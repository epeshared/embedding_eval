#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
from transformers import AutoProcessor
from openvino.runtime import Core

# =========================
# 基本配置
# =========================
hf_id = "/home/xtang/vdb-sandbox/models/embedding/models/openai/clip-vit-base-patch32"
ov_dir = Path("./ov_models/clip-vit-base-patch32")
device_ov = "CPU"

flickr_img_dir = "./datasets/Flickr8k/Flicker8k_Dataset/"
flickr_token_txt = "./datasets/Flickr8k/Flickr8k.token.txt"

rand_batch_size = 100
rand_img_hw = (224, 224)
rand_seed = 42


# =========================
# 工具函数
# =========================
def mean_pooling_np(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[..., None].astype(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(axis=1)
    denom = np.clip(mask.sum(axis=1), 1e-9, None)
    return summed / denom


def l2_normalize_np(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, ord=2, axis=axis, keepdims=True), eps, None)


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
            texts.append(caps[0])
            if limit and len(image_paths) >= limit:
                break
    print(f"✅ Flickr8k 加载样本数：{len(image_paths)}")
    return image_paths, texts


# =========================
# OpenVINO 编码
# =========================
def _compiled_feed(compiled, ids, mask, pixel):
    feed = {}
    for inp in compiled.inputs:
        n = inp.get_any_name().lower()
        if "pixel" in n or "image" in n:
            feed[inp] = pixel
        elif "mask" in n:
            feed[inp] = mask
        elif "input" in n and "id" in n:
            feed[inp] = ids
    return feed


def _extract_embeds(compiled, res, mask):
    img_emb, txt_emb = None, None
    for out in compiled.outputs:
        n = out.get_any_name().lower()
        v = np.array(res[out])
        if "image_embeds" in n:
            img_emb = v
        elif "text_embeds" in n:
            txt_emb = v
    if img_emb is None or txt_emb is None:
        for out in compiled.outputs:
            n = out.get_any_name().lower()
            v = np.array(res[out])
            if img_emb is None and "vision" in n:
                img_emb = v.mean(axis=1)
            if txt_emb is None and "text" in n and "last_hidden_state" in n:
                txt_emb = mean_pooling_np(v, mask)
    return l2_normalize_np(img_emb), l2_normalize_np(txt_emb)


def encode_with_ov(compiled, processor, image_paths, texts, batch_size=32):
    t0 = time.time()
    N = len(image_paths)
    img_all, txt_all = [], []
    for i in range(0, N, batch_size):
        j = min(i + batch_size, N)
        imgs = [Image.open(p).convert("RGB") for p in image_paths[i:j]]
        caps = texts[i:j]
        both = processor(text=caps, images=imgs, return_tensors="np", padding=True)
        pixel = both["pixel_values"].astype("float32")
        ids = both["input_ids"].astype("int64")
        mask = both["attention_mask"].astype("int64")
        feed = _compiled_feed(compiled, ids, mask, pixel)
        res = compiled(feed)
        img, txt = _extract_embeds(compiled, res, mask)
        img_all.append(img)
        txt_all.append(txt)
    img_embs = np.concatenate(img_all)
    txt_embs = np.concatenate(txt_all)
    t1 = time.time()
    print(f"⚙️ 编码完成: {N} 对样本, 耗时 {t1 - t0:.3f}s, 平均每样本 {(t1 - t0) / N * 1000:.2f} ms")
    return img_embs, txt_embs


# =========================
# 检索打印
# =========================
def retrieval_print(img_embs, txt_embs, image_paths, texts, topk=3):
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
# 随机 DEMO
# =========================
def random_demo(processor, compiled, batch_size=8, img_hw=(224, 224), seed=42):
    t_start = time.time()
    np.random.seed(seed)
    np_imgs = [np.random.randint(0, 255, (img_hw[0], img_hw[1], 3), dtype=np.uint8) for _ in range(batch_size)]
    pil_imgs = [Image.fromarray(a) for a in np_imgs]
    texts = [f"random text #{i}" for i in range(batch_size)]

    both = processor(text=texts, images=pil_imgs, return_tensors="np", padding=True)
    pixel = both["pixel_values"].astype("float32")
    ids = both["input_ids"].astype("int64")
    mask = both["attention_mask"].astype("int64")
    feed = _compiled_feed(compiled, ids, mask, pixel)

    t_infer0 = time.time()
    res = compiled(feed)
    t_infer1 = time.time()

    img, txt = _extract_embeds(compiled, res, mask)
    sims = (img @ txt.T)

    t_total = time.time() - t_start
    # print(f"⚙️ DEMO 推理耗时 {t_infer1 - t_infer0:.3f}s，总耗时 {t_total:.3f}s")
    return t_total


# =========================
# 主程序
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP-OpenVINO 图文检索与计时")
    parser.add_argument("--mode", choices=["demo", "flickr"], default="demo",
                        help="demo：随机生成；flickr：用 Flickr8k 实测")
    parser.add_argument("--limit", type=int, default=100, help="flickr 模式下限制样本数")
    parser.add_argument("--loops", type=int, default=1, help="循环次数以便统计平均时间")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(hf_id)
    compiled = load_ov_compiled(ov_dir, device_ov)

    all_times = []

    for loop in range(args.loops):
        # print(f"\n====== 第 {loop+1}/{args.loops} 次运行 ({args.mode}) ======")
        t0 = time.time()

        if args.mode == "demo":
            total_time = random_demo(processor, compiled, rand_batch_size, rand_img_hw, rand_seed)
        else:
            image_paths, texts = load_flickr8k_pairs(flickr_img_dir, flickr_token_txt, limit=args.limit)
            img_embs, txt_embs = encode_with_ov(compiled, processor, image_paths, texts, batch_size=32)
            total_time = time.time() - t0
            retrieval_print(img_embs, txt_embs, image_paths, texts)            
            print(f"🕒 本次 Flickr8k 总耗时: {total_time:.3f}s")

        all_times.append(total_time)

    # 打印统计
    print("\n====== 性能统计 ======")
    # for i, t in enumerate(all_times):
    #     print(f"  第{i+1}次: {t:.3f}s")
    avg_t = sum(all_times) / len(all_times)
    print(f"  平均耗时: {avg_t:.3f}s  （循环 {args.loops} 次）")
    print(f"  总耗时: {sum(all_times):.3f}s ")
