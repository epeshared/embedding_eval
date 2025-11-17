#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import time
import transformers
import asyncio
import concurrent.futures
import argparse
import torch

try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    ipex = None
    IPEX_AVAILABLE = False

# 你可以把这里改成你机器上的实际目录
MODEL_ALIASES = {
    # 小模型
    "clip-base": "/home/xtang/vdb-sandbox/models/embedding/models/openai/clip-vit-base-patch32",
    "openai/clip-vit-base-patch32": "/home/xtang/vdb-sandbox/models/embedding/models/openai/clip-vit-base-patch32",

    # 大模型（336）
    "clip-large-336": "/home/xtang/vdb-sandbox/models/embedding/models/openai/clip-vit-large-patch14-336",
    "openai/clip-vit-large-patch14-336": "/home/xtang/vdb-sandbox/models/embedding/models/openai/clip-vit-large-patch14-336",
}

# 给一个默认，随便选一个
DEFAULT_MODEL = "openai/clip-vit-base-patch32"


def resolve_model_path(model_name: str) -> str:
    """把命令行传进来的名字映射到本地目录或HF名字。"""
    if model_name in MODEL_ALIASES:
        return MODEL_ALIASES[model_name]
    # 没在别名表里，就当它是个本地目录或HF名字
    return model_name


def prepare_huggingface_model(
    pretrained_model_name_or_path: str | os.PathLike,
    device: str = "auto",
    **model_params,
):
    """
    Prepare huggingface model and processor.
    """
    processor = transformers.AutoProcessor.from_pretrained(
        pretrained_model_name_or_path, **model_params
    )

    config = transformers.AutoConfig.from_pretrained(
        pretrained_model_name_or_path, **model_params
    )
    if hasattr(config, "auto_map"):
        class_name = next(
            (k for k in config.auto_map if k.startswith("AutoModel")), "AutoModel"
        )
    else:
        class_name = config.architectures[0]

    model_class = getattr(transformers, class_name)

    if device == "auto":
        # 让 HF 自己决定把模型丢到 GPU 还是 CPU（device_map="auto"）
        model = model_class.from_pretrained(
            pretrained_model_name_or_path, device_map="auto", **model_params
        )
    else:
        model = model_class.from_pretrained(
            pretrained_model_name_or_path, **model_params
        )
        model = model.to(device)

    return (model, processor)


class BenchmarkImage:
    def __init__(self, device: str, data_type: str, model_path: str):
        self.text = "a beautiful landscape"

        # 根据命令行传进来的名字/路径来加载
        self.model, self.processor = prepare_huggingface_model(
            model_path, device=device
        )

        print("Loaded model:", model_path)
        print("IPEX_AVAILABLE=", IPEX_AVAILABLE)

        # dtype 处理
        if data_type == "bf16":
            if IPEX_AVAILABLE:
                self.model = ipex.optimize(
                    self.model, dtype=torch.bfloat16, inplace=True
                )
            self.model.to(torch.bfloat16)
        elif data_type == "fp16":
            if IPEX_AVAILABLE:
                self.model = ipex.optimize(
                    self.model, dtype=torch.float16, inplace=True
                )
            self.model.to(torch.float16)
        # fp32 不动

        # 根据模型猜一下图像尺寸，供下面造随机图用
        # 真正的 resize 还是 processor 来做
        if "336" in model_path or "large-patch14-336" in model_path:
            self.image_h = 336
            self.image_w = 336
        else:
            # base-patch32 一般 224
            self.image_h = 224
            self.image_w = 224

        print("Init Finished! image_size=(", self.image_h, self.image_w, ")")

    def benchmark(self, inputs):
        # 把 inputs 丢到模型所在的 device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        _ = outputs.logits_per_text


async def run_benchmark_async(instance: BenchmarkImage, batch_size, num_iter, executor):
    loop = asyncio.get_event_loop()

    def run_benchmark():
        # 按实例的 image size 造图，这样不同模型切换不会错
        images = [
            np.random.randint(
                0, 255, (instance.image_h, instance.image_w, 3), dtype=np.uint8
            )
            for _ in range(batch_size)
        ]
        inputs = instance.processor(
            text=[instance.text], images=images, return_tensors="pt", padding=True
        )

        for _ in range(num_iter):
            instance.benchmark(inputs)

    return await loop.run_in_executor(executor, run_benchmark)


async def benchmark_image_async(
    parallelism, batch_size, num_iter, device, data_type, model_name
):
    model_path = resolve_model_path(model_name)
    print(
        f"开始异步并行测试: parallelism={parallelism}, batch_size={batch_size}, "
        f"num_iter={num_iter}, device={device}, data_type={data_type}, model={model_path}"
    )

    instances = []
    for _ in range(parallelism):
        inst = BenchmarkImage(device, data_type, model_path)
        instances.append(inst)

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
        tasks = []
        for inst in instances:
            tasks.append(
                run_benchmark_async(inst, batch_size, num_iter, executor)
            )

        start = time.time()
        await asyncio.gather(*tasks)
        total = time.time() - start

        # ====== 这里加 TPS 计算 ======
        total_images = parallelism * batch_size * num_iter
        tps = total_images / total if total > 0 else 0.0

        print(
            f"total time: {total:.4f}s, "
            f"total_images={total_images}, TPS={tps:.2f} images/s"
        )


def benchmark_image(
    parallelism, batch_size, num_iter, device, data_type, model_name
):
    asyncio.run(
        benchmark_image_async(
            parallelism, batch_size, num_iter, device, data_type, model_name
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP model benchmark (random images)")
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_iter", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data_type", type=str, default="fp32")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=(
            "Model name or local path. "
            "Built-ins: 'clip-base', 'openai/clip-vit-base-patch32', "
            "'clip-large-336', 'openai/clip-vit-large-patch14-336'"
        ),
    )
    args = parser.parse_args()

    benchmark_image(
        parallelism=args.parallelism,
        batch_size=args.batch_size,
        num_iter=args.num_iter,
        device=args.device,
        data_type=args.data_type,
        model_name=args.model,
    )
