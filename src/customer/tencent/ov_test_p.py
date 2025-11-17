#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
from transformers import AutoProcessor
from openvino.runtime import Core, AsyncInferQueue

# =========================
# 基本配置（按需修改）
# =========================
hf_id = "/home/xtang/vdb-sandbox/models/embedding/models/openai/clip-vit-base-patch32"
ov_dir = Path("./ov_models/clip-vit-base-patch32")
device_ov = "CPU"

# Flickr8k 路径
flickr_img_dir = "./datasets/Flickr8k/Flicker8k_Dataset/"
flickr_token_txt = "./datasets/Flickr8k/Flickr8k.token.txt"

# 随机 DEMO 参数
rand_batch_size = 200
rand_img_hw = (224, 224)  # (H, W)
rand_seed = 42


# =========================
# 工具函数（NumPy）
# =========================
def mean_pooling_np(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[..., None].astype(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(axis=1)
    denom = np.clip(mask.sum(axis=1), 1e-9, None)
    return summed / denom

def l2_normalize_np(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, ord=2, axis=axis, keepdims=True), eps, None)


# =========================
# OpenVINO Runtime：载入已导出的 IR
# =========================
def load_ov_compiled(ov_dir: Path, device: str = "CPU"):
    xml = ov_dir / "openvino_model.xml"
    bin_ = ov_dir / "openvino_model.bin"
    assert xml.exists() and bin_.exists(), f"在 {ov_dir} 下找不到 openvino_model.xml/.bin"
    core = Core()
    model = core.read_model(xml.as_posix(), bin_.as_posix())
    compiled = core.compile_model(model, device)
    return compiled


# =========================
# Flickr8k 数据加载
# =========================
def load_flickr8k_pairs(image_dir: str, token_txt: str, limit: Optional[int] = None) -> Tuple[List[Path], List[str]]:
    image_dir = Path(image_dir)
    token_file = Path(token_txt)
    assert image_dir.exists() and token_file.exists(), "请检查 Flickr8k 数据路径是否正确。"

    img2caps = {}
    with token_file.open("r", encoding="utf-8") as f:
        for line in f:
            img_tag, caption = line.strip().split("\t")
            img = img_tag.split("#")[0]
            img2caps.setdefault(img, []).append(caption)

    image_paths, texts = [], []
    for img, caps in sorted(img2caps.items()):
        p = image_dir / img
        if p.exists():
            image_paths.append(p)
            texts.append(caps[0])  # 每张图取第 1 条
            if limit and len(image_paths) >= limit:
                break
    print(f"✅ Flickr8k 加载样本数：{len(image_paths)}")
    return image_paths, texts


# =========================
# 推理辅助
# =========================
def _compiled_feed(compiled, ids: np.ndarray, mask: np.ndarray, pixel: np.ndarray):
    feed = {}
    for inp in compiled.inputs:
        n = inp.get_any_name().lower()
        if "pixel" in n or "image" in n:
            feed[inp] = pixel
        elif "mask" in n:
            feed[inp] = mask
        elif "input" in n and "id" in n:
            feed[inp] = ids
    # 兜底（极少数模型名字不含关键词）
    if len(feed) < 3:
        for inp in compiled.inputs:
            if inp not in feed:
                shp = tuple(inp.get_shape())
                if len(shp) == 4:
                    feed[inp] = pixel
                elif len(shp) == 2 and shp[1] == ids.shape[1]:
                    if "mask" in inp.get_any_name().lower():
                        feed[inp] = mask
                    else:
                        feed[inp] = ids
    return feed

def _extract_embeds(compiled, res_dict, mask: np.ndarray):
    img_emb = txt_emb = None
    lh_vis = lh_text = lh_common = None

    for out in compiled.outputs:
        n = out.get_any_name().lower()
        v = np.array(res_dict[out])
        if "image_embeds" in n:
            img_emb = v
        elif "text_embeds" in n:
            txt_emb = v
        elif "vision" in n and "last_hidden_state" in n:
            lh_vis = v
        elif "text" in n and "last_hidden_state" in n:
            lh_text = v
        elif "last_hidden_state" in n:
            lh_common = v

    if img_emb is None:
        if lh_vis is not None:
            img_emb = lh_vis.mean(axis=1)
        elif lh_common is not None:
            img_emb = lh_common.mean(axis=1)
        else:
            raise RuntimeError("无法从输出中获取图像特征。")

    if txt_emb is None:
        if lh_text is not None:
            txt_emb = mean_pooling_np(lh_text, mask)
        elif lh_common is not None:
            txt_emb = mean_pooling_np(lh_common, mask)
        else:
            raise RuntimeError("无法从输出中获取文本特征。")

    return l2_normalize_np(img_emb), l2_normalize_np(txt_emb)


# =========================
# 单线程编码
# =========================
def encode_with_ov(compiled, processor, image_paths: List[Path], texts: List[str], batch_size=32):
    t0 = time.time()
    N = len(image_paths)
    img_all, txt_all = [], []
    for i in range(0, N, batch_size):
        j = min(i + batch_size, N)
        imgs = [Image.open(p).convert("RGB") for p in image_paths[i:j]]
        caps = texts[i:j]
        both = processor(text=caps, images=imgs, return_tensors="np", padding=True)
        pixel = both["pixel_values"].astype("float32")
        ids   = both["input_ids"].astype("int64")
        mask  = both["attention_mask"].astype("int64")
        if pixel.ndim == 4 and pixel.shape[1] not in (1, 3) and pixel.shape[-1] in (1, 3):
            pixel = np.transpose(pixel, (0, 3, 1, 2)).copy()
        feed = _compiled_feed(compiled, ids, mask, pixel)
        res = compiled(feed)
        img, txt = _extract_embeds(compiled, res, mask)
        img_all.append(img)
        txt_all.append(txt)
    img_embs = np.concatenate(img_all)
    txt_embs = np.concatenate(txt_all)
    t1 = time.time()
    print(f"⚙️ 编码完成: {N} 对样本, 耗时 {t1 - t0:.3f}s, 平均 {(t1 - t0)/N*1000:.2f} ms/样本")
    return img_embs, txt_embs


# =========================
# 多线程编码（线程池 + InferRequest/线程）
# =========================
def _encode_shard_with_infer_request(
    compiled,
    processor,
    image_paths: List[Path],
    texts: List[str],
    start: int,
    end: int,
    batch_size: int = 32,
):
    infer_request = compiled.create_infer_request()
    img_chunks, txt_chunks = [], []
    for i in range(start, end, batch_size):
        j = min(i + batch_size, end)
        imgs = [Image.open(p).convert("RGB") for p in image_paths[i:j]]
        caps = texts[i:j]
        both = processor(text=caps, images=imgs, return_tensors="np", padding=True)
        pixel = both["pixel_values"].astype("float32", copy=False)
        ids   = both["input_ids"].astype("int64",   copy=False)
        mask  = both["attention_mask"].astype("int64", copy=False)
        if pixel.ndim == 4 and pixel.shape[1] not in (1, 3) and pixel.shape[-1] in (1, 3):
            pixel = np.transpose(pixel, (0, 3, 1, 2)).copy()
        feed = _compiled_feed(compiled, ids, mask, pixel)
        res  = infer_request.infer(feed)
        img_emb, txt_emb = _extract_embeds(compiled, res, mask)
        img_chunks.append(img_emb)
        txt_chunks.append(txt_emb)
    return np.concatenate(img_chunks, axis=0), np.concatenate(txt_chunks, axis=0)

def encode_with_ov_parallel(
    compiled,
    processor,
    image_paths: List[Path],
    texts: List[str],
    batch_size: int = 32,
    workers: int = 2,
):
    N = len(image_paths)
    workers = max(1, int(workers))
    if workers == 1 or N <= batch_size:
        return encode_with_ov(compiled, processor, image_paths, texts, batch_size=batch_size)

    # 切分样本
    shard_ranges = []
    per = math.ceil(N / workers)
    for k in range(workers):
        s = k * per
        e = min(N, (k + 1) * per)
        if s < e:
            shard_ranges.append((s, e))

    img_parts, txt_parts = [None] * len(shard_ranges), [None] * len(shard_ranges)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for idx, (s, e) in enumerate(shard_ranges):
            fut = ex.submit(
                _encode_shard_with_infer_request,
                compiled, processor, image_paths, texts, s, e, batch_size
            )
            futures[fut] = idx
        for fut in as_completed(futures):
            idx = futures[fut]
            img_shard, txt_shard = fut.result()
            img_parts[idx] = img_shard
            txt_parts[idx] = txt_shard

    img_embs = np.concatenate(img_parts, axis=0)
    txt_embs = np.concatenate(txt_parts, axis=0)
    t1 = time.time()
    print(f"⚙️ 并行编码完成: {N} 对, 线程 {workers} 个, 耗时 {t1 - t0:.3f}s, 平均 {(t1 - t0)/N*1000:.2f} ms/样本")
    return img_embs, txt_embs


# =========================
# AsyncInferQueue 异步编码
# =========================
def encode_with_ov_async(
    compiled,
    processor,
    image_paths: List[Path],
    texts: List[str],
    batch_size: int = 32,
    workers: int = 2,
):
    """
    用 OpenVINO AsyncInferQueue 做多请求异步推理
    """
    N = len(image_paths)
    if N == 0:
        return np.zeros((0, 1)), np.zeros((0, 1))
    workers = max(1, int(workers))

    # 1) 预处理 -> feed 列表
    batches = []
    for i in range(0, N, batch_size):
        j = min(i + batch_size, N)
        imgs = [Image.open(p).convert("RGB") for p in image_paths[i:j]]
        caps = texts[i:j]
        both = processor(text=caps, images=imgs, return_tensors="np", padding=True)
        pixel = both["pixel_values"].astype("float32", copy=False)
        ids   = both["input_ids"].astype("int64",   copy=False)
        mask  = both["attention_mask"].astype("int64", copy=False)
        if pixel.ndim == 4 and pixel.shape[1] not in (1, 3) and pixel.shape[-1] in (1, 3):
            pixel = np.transpose(pixel, (0, 3, 1, 2)).copy()
        feed = _compiled_feed(compiled, ids, mask, pixel)
        batches.append((feed, mask))

    num_batches = len(batches)
    img_parts = [None] * num_batches
    txt_parts = [None] * num_batches

    # 2) AIQ + 回调
    aiq = AsyncInferQueue(compiled, jobs=workers)

    def _cb(request, userdata):
        bidx, mask_np = userdata  # 哪个 batch
        res = request.results
        img_emb, txt_emb = _extract_embeds(compiled, res, mask_np)
        img_parts[bidx] = img_emb
        txt_parts[bidx] = txt_emb

    aiq.set_callback(_cb)

    # 3) 启动 + 等待
    t0 = time.time()
    for bidx, (feed, mask) in enumerate(batches):
        aiq.start_async(feed, userdata=(bidx, mask))
    aiq.wait_all()
    t1 = time.time()

    img_embs = l2_normalize_np(np.concatenate(img_parts, axis=0), axis=-1)
    txt_embs = l2_normalize_np(np.concatenate(txt_parts, axis=0), axis=-1)
    print(f"⚙️ AsyncInferQueue 编码完成: {N} 对, 队列 {workers}, 耗时 {t1 - t0:.3f}s, 平均 {(t1 - t0)/N*1000:.2f} ms/样本")
    return img_embs, txt_embs


# =========================
# 检索打印
# =========================
def retrieval_print(img_embs: np.ndarray, txt_embs: np.ndarray, image_paths: List[Path], texts: List[str], topk=3):
    sims_t2i = txt_embs @ img_embs.T
    sims_i2t = img_embs @ txt_embs.T

    print("\n===== 图像 → 文本 (Image → Text) =====")
    for i in range(min(3, len(image_paths))):
        idx = np.argsort(-sims_i2t[i])[:topk]
        print(f"[Image {i}] {image_paths[i].name}")
        for r, j in enumerate(idx, 1):
            print(f"  {r}. '{texts[j]}'  sim={sims_i2t[i, j]:.4f}")

    print("\n===== 文本 → 图像 (Text → Image) =====")
    for i in range(min(3, len(texts))):
        idx = np.argsort(-sims_t2i[i])[:topk]
        print(f"[Text {i}] '{texts[i]}'")
        for r, j in enumerate(idx, 1):
            print(f"  {r}. {image_paths[j].name}  sim={sims_t2i[i, j]:.4f}")


# =========================
# 随机 DEMO（单线程 / AIQ）
# =========================
def random_demo(processor, compiled, batch_size=8, img_hw=(224, 224), seed=42):
    t_start = time.time()
    np.random.seed(seed)
    np_imgs = [np.random.randint(0, 255, (img_hw[0], img_hw[1], 3), dtype=np.uint8) for _ in range(batch_size)]
    pil_imgs = [Image.fromarray(a) for a in np_imgs]
    texts = [f"random text #{i}" for i in range(batch_size)]

    both = processor(text=texts, images=pil_imgs, return_tensors="np", padding=True)
    pixel = both["pixel_values"].astype("float32")
    ids   = both["input_ids"].astype("int64")
    mask  = both["attention_mask"].astype("int64")
    if pixel.ndim == 4 and pixel.shape[1] not in (1, 3) and pixel.shape[-1] in (1, 3):
        pixel = np.transpose(pixel, (0, 3, 1, 2)).copy()
    feed  = _compiled_feed(compiled, ids, mask, pixel)

    t_infer0 = time.time()
    res = compiled(feed)
    t_infer1 = time.time()

    img, txt = _extract_embeds(compiled, res, mask)
    sims = (img @ txt.T)

    t_total = time.time() - t_start
    print(f"⚙️ DEMO 推理耗时 {t_infer1 - t_infer0:.3f}s，总耗时 {t_total:.3f}s")
    for i in range(min(3, batch_size)):
        idx = np.argsort(-sims[i])[:3]
        print(f"[Rand Image #{i}] top-3 text matches:", ", ".join([f"{j}:{sims[i,j]:.3f}" for j in idx]))
    return t_total

def random_demo_async(processor, compiled, batch_size=8, img_hw=(224,224), seed=42, workers=2):
    np.random.seed(seed)
    np_imgs = [np.random.randint(0, 255, (img_hw[0], img_hw[1], 3), dtype=np.uint8) for _ in range(batch_size)]
    pil_imgs = [Image.fromarray(a) for a in np_imgs]
    texts = [f"random text #{i}" for i in range(batch_size)]

    both = processor(text=texts, images=pil_imgs, return_tensors="np", padding=True)
    pixel = both["pixel_values"].astype("float32")
    ids   = both["input_ids"].astype("int64")
    mask  = both["attention_mask"].astype("int64")
    feed  = _compiled_feed(compiled, ids, mask, pixel)

    aiq = AsyncInferQueue(compiled, jobs=workers)
    holder = [None] * workers

    def _cb(req, userdata):
        holder[userdata] = req.results

    aiq.set_callback(_cb)

    t0 = time.time()
    # 把 batch 切分为 workers 段
    parts = np.array_split(np.arange(batch_size), workers)
    for idx, ids in enumerate(parts):
        sub_feed = {k: (v[ids] if hasattr(v, "__getitem__") and len(v) == batch_size else v)
                    for k, v in feed.items()}
        aiq.start_async(sub_feed, userdata=idx)

    aiq.wait_all()
    t1 = time.time()

    # 聚合结果
    all_img, all_txt = [], []
    for res in holder:
        if res is not None:
            img, txt = _extract_embeds(compiled, res, mask)
            all_img.append(img)
            all_txt.append(txt)
    img = l2_normalize_np(np.concatenate(all_img), axis=-1)
    txt = l2_normalize_np(np.concatenate(all_txt), axis=-1)
    sims = img @ txt.T

    print(f"⚙️ DEMO (AsyncInferQueue True Parallel) 耗时 {t1 - t0:.3f}s，batch={batch_size}, workers={workers}")
    for i in range(min(3, batch_size)):
        idx = np.argsort(-sims[i])[:3]
        print(f"[Rand Image #{i}] top-3 text matches:",
              ", ".join([f"{j}:{sims[i,j]:.3f}" for j in idx]))
    return t1 - t0



# =========================
# 主程序
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP-OpenVINO 图文检索 / 计时 / 并行 (threads/async)")
    parser.add_argument("--mode", choices=["demo", "flickr"], default="demo",
                        help="demo：随机；flickr：用 Flickr8k 实测")
    parser.add_argument("--limit", type=int, default=100, help="flickr 模式下限制样本数")
    parser.add_argument("--loops", type=int, default=1, help="循环次数做平均")
    parser.add_argument("--workers", type=int, default=1, help="并行度：threads 的线程数或 async 的队列大小")
    parser.add_argument("--batch-size", type=int, default=32, help="编码 batch size（flickr 模式）")
    parser.add_argument("--engine", choices=["threads", "async"], default="threads",
                        help="并行引擎：threads=ThreadPoolExecutor，async=OpenVINO AsyncInferQueue")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(hf_id)
    compiled = load_ov_compiled(ov_dir, device_ov)

    all_times = []

    for loop in range(args.loops):
        print(f"\n====== 第 {loop+1}/{args.loops} 次运行 ({args.mode}, engine={args.engine}, workers={args.workers}) ======")
        t0 = time.time()

        if args.mode == "demo":
            if args.engine == "async":
                total_time = random_demo_async(processor, compiled,
                                               batch_size=rand_batch_size,
                                               img_hw=rand_img_hw,
                                               seed=rand_seed,
                                               workers=args.workers)
            else:
                if args.workers > 1:
                    # 线程池版 demo：把 batch 切片并行（简单演示）
                    # 为保持脚本简洁，这里直接复用单线程 demo 的总时间作为参考
                    total_time = random_demo(processor, compiled,
                                             batch_size=rand_batch_size,
                                             img_hw=rand_img_hw,
                                             seed=rand_seed)
                else:
                    total_time = random_demo(processor, compiled,
                                             batch_size=rand_batch_size,
                                             img_hw=rand_img_hw,
                                             seed=rand_seed)
        else:
            image_paths, texts = load_flickr8k_pairs(flickr_img_dir, flickr_token_txt, limit=args.limit)
            if args.engine == "async":
                img_embs, txt_embs = encode_with_ov_async(compiled, processor, image_paths, texts,
                                                          batch_size=args.batch_size, workers=args.workers)
            else:
                if args.workers > 1:
                    img_embs, txt_embs = encode_with_ov_parallel(compiled, processor, image_paths, texts,
                                                                 batch_size=args.batch_size, workers=args.workers)
                else:
                    img_embs, txt_embs = encode_with_ov(compiled, processor, image_paths, texts,
                                                        batch_size=args.batch_size)
            retrieval_print(img_embs, txt_embs, image_paths, texts)
            total_time = time.time() - t0
            print(f"🕒 本次 Flickr8k 总耗时: {total_time:.3f}s")

        all_times.append(total_time)

    # 统计
    print("\n====== 性能统计 ======")
    for i, t in enumerate(all_times):
        print(f"  第{i+1}次: {t:.3f}s")
    avg_t = sum(all_times) / len(all_times)
    print(f"  平均耗时: {avg_t:.3f}s  （循环 {args.loops} 次, engine={args.engine}, workers={args.workers}）")
