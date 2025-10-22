#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import time
import asyncio
import concurrent.futures
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import transformers

# ===== Optional IPEX =====
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    ipex = None
    IPEX_AVAILABLE = False

# ===== PIL for Flickr8k images =====
try:
    from PIL import Image
    PIL_OK = True
except Exception:
    Image = None
    PIL_OK = False


# -----------------------------
# HF model & processor loading
# -----------------------------
def prepare_huggingface_model(
    pretrained_model_name_or_path: str | os.PathLike, **model_params
):
    """
    Prepare huggingface model and processor.
    兼容根据 config.auto_map 或 architectures 选择模型类。
    """
    processor = transformers.AutoProcessor.from_pretrained(
        pretrained_model_name_or_path, **model_params
    )

    config = transformers.AutoConfig.from_pretrained(
        pretrained_model_name_or_path, **model_params
    )
    if hasattr(config, "auto_map") and isinstance(config.auto_map, dict):
        # 选第一个以 "AutoModel" 开头的键
        class_name = next((k for k in config.auto_map if k.startswith("AutoModel")), "AutoModel")
        model_class = getattr(transformers, class_name)
    else:
        # 回退：用 architectures[0] 对应的类名
        arch = getattr(config, "architectures", None)
        if arch and len(arch) > 0:
            class_name = arch[0]
            model_class = getattr(transformers, class_name)
        else:
            # 最保守的回退
            model_class = transformers.AutoModel

    model = model_class.from_pretrained(pretrained_model_name_or_path, **model_params)
    return (model, processor)


# -----------------------------
# Flickr8k helpers
# -----------------------------
def _read_flickr8k_captions(captions_file: str) -> Dict[str, List[str]]:
    """读取 Flickr8k.token.txt -> {filename: [captions...]}"""
    mapping: Dict[str, List[str]] = {}
    with open(captions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            left, cap = line.split("\t", 1)
            img, _ = left.split("#", 1)
            mapping.setdefault(img, []).append(cap)
    return mapping


def _load_flickr8k_samples(
    images_dir: str,
    captions_file: str,
    total: int,
    pick_caption: str = "first",
) -> Tuple[List[np.ndarray], List[str]]:
    """
    返回长度为 total 的 (images, texts)，images 为 RGB uint8 HWC。
    total 可大于目录图片数，会循环取。
    """
    if not PIL_OK:
        raise RuntimeError("PIL (Pillow) 未安装，无法读取 Flickr8k 图片。请先 `pip install pillow`。")

    cap_map = _read_flickr8k_captions(captions_file)
    filenames = [
        fn for fn in os.listdir(images_dir)
        if fn.lower().endswith((".jpg", ".jpeg", ".png")) and fn in cap_map
    ]
    if not filenames:
        raise RuntimeError(f"在 {images_dir} 未发现与 {os.path.basename(captions_file)} 匹配的图片文件。")

    filenames.sort()
    images: List[np.ndarray] = []
    texts: List[str] = []
    n = len(filenames)

    rng = np.random.default_rng()

    for i in range(total):
        fn = filenames[i % n]
        img_path = os.path.join(images_dir, fn)
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            arr = np.array(im, dtype=np.uint8)
        caps = cap_map[fn]
        cap = caps[rng.integers(0, len(caps))] if pick_caption == "random" else caps[0]
        images.append(arr)
        texts.append(cap)

    return images, texts


# -----------------------------
# BatchFeature slicing
# -----------------------------
def _slice_batchfeature(proto_bf: transformers.BatchFeature, start: int, end: int) -> transformers.BatchFeature:
    """从 BatchFeature 中切出 [start:end) 的一个 batch，并保持其为 BatchFeature（带 .to()）。"""
    data = {}
    for k, v in proto_bf.items():
        if torch.is_tensor(v):
            data[k] = v[start:end]
        elif isinstance(v, (list, tuple)):
            data[k] = v[start:end]
        else:
            data[k] = v
    return transformers.BatchFeature(
        data=data,
        tensor_type=getattr(proto_bf, "tensor_type", None)
    )


# -----------------------------
# Benchmark runner
# -----------------------------
class BenchmarkImage:
    def __init__(self, device: str, data_type: str):
        self.text = "a beautiful landscape"
        # 你本地的 CLIP 权重路径
        self.model, self.processor = prepare_huggingface_model(
            "/home/xtang/vdb-sandbox/models/embedding/models/openai/clip-vit-base-patch32",
            device_map=device,
        )
        print(f"IPEX_AVAILABLE={IPEX_AVAILABLE}")
        dt = (data_type or "fp32").lower()
        if dt == "bf16":
            if IPEX_AVAILABLE:
                self.model = ipex.optimize(self.model, dtype=torch.bfloat16, inplace=True)
            self.model.to(torch.bfloat16)
        elif dt == "fp16":
            if IPEX_AVAILABLE:
                self.model = ipex.optimize(self.model, dtype=torch.float16, inplace=True)
            self.model.to(torch.float16)
        else:
            # 默认 fp32，不额外处理
            pass

        # 如需 torch.compile，可按需开启
        # self.model = torch.compile(self.model)

        print("Init Finished!")

    def benchmark(self, inputs: transformers.BatchFeature):
        # 每次调用都会尝试把 inputs 搬到模型设备（BatchFeature.to 为就地操作）
        inputs.to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        _ = getattr(outputs, "logits_per_text", None)  # 访问一下，避免优化器去掉


# -----------------------------
# 外置化：一次性准备数据 + processor
# -----------------------------
def _prepare_inputs_once(
    instance: BenchmarkImage,
    batch_size: int,
    num_iter: int,
    data_source: str,
    flickr_images_dir: Optional[str],
    flickr_captions_file: Optional[str],
    flickr_caption_pick: str,
    max_samples: Optional[int],
) -> Tuple[Optional[transformers.BatchFeature], int, int]:
    """
    全局一次性准备 (images, texts) 并构造 big_inputs（CPU 上）。
    返回: (big_inputs, total_samples_aligned, num_batches)
    """
    # 全局目标数据量（你的原始定义：不乘 parallelism；随后按 batch 分摊给各线程）
    desired_total = num_iter * batch_size

    if data_source == "flickr8k":
        if not flickr_images_dir or not flickr_captions_file:
            raise ValueError("--data_source=flickr8k 需要同时提供 --flickr_images_dir 与 --flickr_captions_file")
        total = min(desired_total, max_samples) if (max_samples and max_samples > 0) else desired_total
        total = (total // batch_size) * batch_size  # 向下对齐到 batch
        if total == 0:
            print("[global] 注意：样本量不足一个 batch，跳过。")
            return None, 0, 0
        images, texts = _load_flickr8k_samples(
            flickr_images_dir, flickr_captions_file, total, pick_caption=flickr_caption_pick
        )
    else:
        total = desired_total
        total = (total // batch_size) * batch_size
        if total == 0:
            print("[global] 注意：样本量不足一个 batch，跳过。")
            return None, 0, 0
        images = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(total)
        ]
        texts = [instance.text] * total

    # —— 只在 CPU 上做一次 processor（构造大 BatchFeature）——
    big_inputs = instance.processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True
    )
    num_batches = total // batch_size
    return big_inputs, total, num_batches


# -----------------------------
# 线程工作：仅执行计算，数据切片由主线程传入
# -----------------------------
async def run_benchmark_async(
    instance: BenchmarkImage,
    executor: concurrent.futures.Executor,
    big_inputs: transformers.BatchFeature,
    shard_start: int,          # 该线程切片的起始样本索引（按样本数，而非 batch 序号）
    shard_batches: int,        # 该线程要处理的 batch 数
    batch_size: int,
    copy_per_iter: bool = False,
) -> Tuple[float, int]:
    loop = asyncio.get_event_loop()

    def run_benchmark():
        if shard_batches == 0:
            return 0.0, 0

        is_cuda = hasattr(instance.model, "device") and str(instance.model.device).startswith("cuda")
        start_loop_t = time.time()

        if not copy_per_iter:
            # 只切第一组 batch，循环复用（首次 .to() 后保持在模型设备）
            first = _slice_batchfeature(big_inputs, shard_start, shard_start + batch_size)
            for _ in range(shard_batches):
                instance.benchmark(first)
        else:
            # 每轮取新的切片，触发每轮 H→D copy
            base = shard_start
            for i in range(shard_batches):
                s = base + i * batch_size
                e = s + batch_size
                this_batch = _slice_batchfeature(big_inputs, s, e)
                instance.benchmark(this_batch)

        if is_cuda:
            torch.cuda.synchronize()

        loop_elapsed = time.time() - start_loop_t
        processed = shard_batches * batch_size
        print(f"[worker] for-loop time only: {loop_elapsed:.6f}s, processed={processed}")
        return loop_elapsed, processed

    return await loop.run_in_executor(executor, run_benchmark)


# -----------------------------
# Orchestration
# -----------------------------
async def benchmark_image_async(
    parallelism: int,
    batch_size: int,
    num_iter: int,
    device: str,
    data_type: str,
    copy_per_iter: bool = False,
    data_source: str = "random",
    flickr_images_dir: Optional[str] = None,
    flickr_captions_file: Optional[str] = None,
    flickr_caption_pick: str = "first",
    max_samples: Optional[int] = None,
):
    print(
        f"开始异步并行测试: parallelism={parallelism}, batch_size={batch_size}, num_iter={num_iter}, "
        f"device={device}, data_type={data_type}, copy_per_iter={copy_per_iter}, data_source={data_source}, "
        f"max_samples={max_samples}"
    )
    assert parallelism >= 1, "parallelism 必须 >= 1"

    # 初始化每个线程的实例（各自一份模型，以避免跨线程竞争）
    benchmark_image = [BenchmarkImage(device, data_type) for _ in range(parallelism)]

    # —— 主线程：一次性准备数据 & big_inputs（只做一次 processor）——
    big_inputs, total, num_batches = _prepare_inputs_once(
        benchmark_image[0], batch_size, num_iter*parallelism, data_source,
        flickr_images_dir, flickr_captions_file, flickr_caption_pick, max_samples
    )
    if big_inputs is None or num_batches == 0:
        print("[global] 无可用数据，直接结束。")
        return

    # —— 按 batch 等分切给每个线程 ——（尽量平均：前 extra 个线程多 1 个 batch）
    base_batches = num_batches // parallelism
    extra = num_batches % parallelism

    plan: List[Tuple[int, int]] = []  # (shard_start_sample_index, shard_batches)
    cursor_batches = 0
    for i in range(parallelism):
        bi = base_batches + (1 if i < extra else 0)
        si = cursor_batches * batch_size
        plan.append((si, bi))
        cursor_batches += bi

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
        tasks = []
        for i in range(parallelism):
            shard_start, shard_batches = plan[i]
            if shard_batches == 0:
                continue  # batch 不足时，可能有线程拿不到任务
            task = run_benchmark_async(
                benchmark_image[i],
                executor,
                big_inputs=big_inputs,
                shard_start=shard_start,
                shard_batches=shard_batches,
                batch_size=batch_size,
                copy_per_iter=copy_per_iter,
            )
            tasks.append(task)

        wall_start = time.time()
        per_results = await asyncio.gather(*tasks)
        wall_total = time.time() - wall_start

    # —— 汇总 —— 
    per_loop_times = [r[0] for r in per_results]
    per_processed  = [r[1] for r in per_results]
    total_processed = sum(per_processed)
    avg_loop_time = (sum(per_loop_times) / len(per_loop_times)) if per_loop_times else 0.0

    print("\n==== Loop-time & processed per worker ====")
    for idx, (t, n) in enumerate(zip(per_loop_times, per_processed)):
        print(f"worker-{idx}: loop_time={t:.6f}s, processed={n}")

    overall_tps = (total_processed / avg_loop_time) if avg_loop_time > 0 else 0.0

    print("\n==== Summary ====")
    print(f"global total processed (aligned): {total_processed}  (target: {num_iter * batch_size})")
    print(f"num_batches(total / bs): {num_batches}")
    print(f"avg(loop_time per worker)   (s): {avg_loop_time:.6f}")
    print(f"wall_time_all_workers_done (s): {wall_total:.6f}")
    print(f"overall TPS (processed / avg_loop_time): {overall_tps:.3f}")


def benchmark_image(
    parallelism: int,
    batch_size: int,
    num_iter: int,
    device: str,
    data_type: str,
    copy_per_iter: bool = False,
    data_source: str = "random",
    flickr_images_dir: Optional[str] = None,
    flickr_captions_file: Optional[str] = None,
    flickr_caption_pick: str = "first",
    max_samples: Optional[int] = None,
):
    asyncio.run(
        benchmark_image_async(
            parallelism, batch_size, num_iter, device, data_type,
            copy_per_iter=copy_per_iter, data_source=data_source,
            flickr_images_dir=flickr_images_dir, flickr_captions_file=flickr_captions_file,
            flickr_caption_pick=flickr_caption_pick, max_samples=max_samples
        )
    )


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP model with random data or Flickr8k")
    parser.add_argument("--parallelism", type=int, default=1, help="Parallelism of the benchmark")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size of the model")
    parser.add_argument("--num_iter", type=int, default=5, help="Number of iterations of the model")
    parser.add_argument("--device", type=str, default="auto", help="Target device or device_map (e.g., 'cuda', 'cpu', 'auto')")
    parser.add_argument("--data_type", type=str, default="fp32", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--copy_per_iter", action="store_true",
                        help="每轮都强制 CPU→GPU 拷贝（每轮取新的数据切片）")

    # 数据源 & Flickr8k 路径
    parser.add_argument("--data_source", type=str, default="random", choices=["random", "flickr8k"],
                        help="数据来源：random 或 flickr8k")
    parser.add_argument("--flickr_images_dir", type=str, default=None,
                        help="Flickr8k 图片目录（如 .../Flicker8k_Dataset）")
    parser.add_argument("--flickr_captions_file", type=str, default=None,
                        help="Flickr8k captions 文件（如 .../Flickr8k_text/Flickr8k.token.txt）")
    parser.add_argument("--flickr_caption_pick", type=str, default="first", choices=["first", "random"],
                        help="每张图选哪条 caption（first|random）")

    # 仅对 Flickr8k 生效的最大样本数（按 batch_size 向下对齐）
    parser.add_argument("--max_samples", type=int, default=None,
                        help="仅对 flickr8k 生效：最多读取的样本数（会按 batch_size 向下对齐）")

    args = parser.parse_args()

    benchmark_image(
        parallelism=args.parallelism,
        batch_size=args.batch_size,
        num_iter=args.num_iter,
        device=args.device,
        data_type=args.data_type,
        copy_per_iter=args.copy_per_iter,
        data_source=args.data_source,
        flickr_images_dir=args.flickr_images_dir,
        flickr_captions_file=args.flickr_captions_file,
        flickr_caption_pick=args.flickr_caption_pick,
        max_samples=args.max_samples,
    )
