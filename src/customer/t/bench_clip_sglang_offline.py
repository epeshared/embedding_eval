#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import argparse
import asyncio
import concurrent.futures
from typing import List, Optional

import numpy as np
import torch
import dataclasses
from PIL import Image

import sglang as sgl
from sglang.srt.server_args import ServerArgs


# ====== 模型路径别名（按你的环境改） ======
MODEL_ALIASES = {
    # 小模型
    "clip-base": "/home/xtang/vdb-sandbox/models/embedding/models/openai/clip-vit-base-patch32",
    "openai/clip-vit-base-patch32": "/home/xtang/vdb-sandbox/models/embedding/models/openai/clip-vit-base-patch32",

    # 大模型（336）
    "clip-large-336": "/home/xtang/vdb-sandbox/models/embedding/models/openai/clip-vit-large-patch14-336",
    "openai/clip-vit-large-patch14-336": "/home/xtang/vdb-sandbox/models/embedding/models/openai/clip-vit-large-patch14-336",
}

DEFAULT_MODEL = "openai/clip-vit-base-patch32"


def resolve_model_path(model_name: str) -> str:
    """把命令行传进来的名字映射到本地目录或原样返回。"""
    if model_name in MODEL_ALIASES:
        return MODEL_ALIASES[model_name]
    return model_name


def map_data_type_to_dtype_str(data_type: str) -> str:
    """命令行 data_type -> sglang ServerArgs.dtype 字符串。"""
    dt = data_type.lower()
    if dt in ["bf16", "bfloat16"]:
        return "bfloat16"
    if dt in ["fp16", "half"]:
        return "float16"
    # 默认当作 fp32
    return "float32"


class SGLangEmbeddingBench:
    """使用 sglang.Engine (offline) 做 embedding 基准测试（支持 text / text+image）。"""

    def __init__(
        self,
        device: str,
        data_type: str,
        model_path: str,
        embed_mode: str = "multimodal",
        attention_backend: Optional[str] = None,
        enable_torch_compile: bool = False,
    ):
        """
        embed_mode: "text" 或 "multimodal"
            - text:       只做文本 embedding
            - multimodal: 文本 + 图片 一起喂到 encode
        """
        self.text = "a beautiful landscape"
        self.embed_mode = embed_mode

        # ======== dtype / device 处理 ========
        dtype_str = map_data_type_to_dtype_str(data_type)

        if device == "auto":
            device_str = "auto"
        else:
            device_str = device

        # ======== 组装 ServerArgs ========
        server_args = ServerArgs(
            model_path=model_path,
            dtype=dtype_str,
            device=device_str,
            tp_size=1,
            dp_size=1,
            is_embedding=True,
            enable_torch_compile=enable_torch_compile,
            attention_backend=attention_backend,
            torch_compile_max_bs=16,
            log_level="error",
        )

        engine_kwargs = dataclasses.asdict(server_args)
        print("[Init:sglang] Engine kwargs:", engine_kwargs)

        # ======== 纯 offline Engine（无 HTTP server） ========
        self.engine = sgl.Engine(**engine_kwargs)

        # ======== 根据模型名猜图像尺寸（和你原脚本逻辑一致） ========
        if "336" in model_path or "large-patch14-336" in model_path:
            self.image_h = 336
            self.image_w = 336
        else:
            self.image_h = 224
            self.image_w = 224

        print(
            f"Init Finished! image_size=({self.image_h}, {self.image_w}), "
            f"embed_mode={self.embed_mode}"
        )

    def make_random_images(self, batch_size: int) -> List[np.ndarray]:
        """造一批随机图像，用于压测."""
        images = [
            np.random.randint(
                0, 255, (self.image_h, self.image_w, 3), dtype=np.uint8
            )
            for _ in range(batch_size)
        ]
        return images

    def run_once(self, texts: List[str], images: Optional[List[np.ndarray]] = None):
        """用给定的 texts / images 跑一次 embedding。"""
        if self.embed_mode == "text":
            # 纯文本 embedding：忽略 images
            outputs = self.engine.encode(prompt=texts)
            _ = outputs
        elif self.embed_mode == "multimodal":
            assert images is not None, "multimodal 模式下必须提供 images"
            # 把 numpy 转成 PIL.Image，符合 image_data 的预期输入
            pil_images = [Image.fromarray(img) for img in images]

            # image_data 的典型形式：
            #   - 单张图：Image / path / URL / base64
            #   - batch：list[Image] 或 list[path]
            #   - 多图 per sample：list[list[Image]]
            #
            # 这里我们每条样本 1 张图：list[list[Image]]
            image_data = [[im] for im in pil_images]

            outputs = self.engine.encode(
                prompt=texts,
                image_data=image_data,
            )
            _ = outputs
        else:
            raise ValueError(f"Unsupported embed_mode: {self.embed_mode}")


async def run_benchmark_async(
    instance: SGLangEmbeddingBench, batch_size: int, num_iter: int, executor
):
    """
    对齐你以前的 HF 版本：
      - 在这个函数里显式生成 text / image 输入
      - 然后调用 instance.run_once(...)
    """
    loop = asyncio.get_event_loop()

    def run_benchmark():
        # 确保当前线程有 event loop，给 Engine.encode 内部用
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)

        for _ in range(num_iter):
            if instance.embed_mode == "text":
                # 纯文本：只造 texts
                texts = [instance.text] * batch_size
                images = None
            else:
                # multimodal：造 texts + 随机 images
                images = [
                    np.random.randint(
                        0,
                        255,
                        (instance.image_h, instance.image_w, 3),
                        dtype=np.uint8,
                    )
                    for _ in range(batch_size)
                ]
                texts = [instance.text] * batch_size

            instance.run_once(texts, images)

    return await loop.run_in_executor(executor, run_benchmark)


async def benchmark_embedding_async(
    parallelism: int,
    batch_size: int,
    num_iter: int,
    device: str,
    data_type: str,
    model_name: str,
    embed_mode: str,
):
    model_path = resolve_model_path(model_name)
    print(
        f"开始异步并行测试: parallelism={parallelism}, batch_size={batch_size}, "
        f"num_iter={num_iter}, device={device}, data_type={data_type}, "
        f"embed_mode={embed_mode}, model={model_path}"
    )

    instances: List[SGLangEmbeddingBench] = []
    for _ in range(parallelism):
        if device == "cuda":
            inst = SGLangEmbeddingBench(
                device=device,
                data_type=data_type,
                model_path=model_path,
                embed_mode=embed_mode,
                attention_backend=None,      # 比如 "flashinfer"，按需再开
                enable_torch_compile=False,  # GPU 可以考虑 True，这里先关
            )
        elif device == "cpu":
            inst = SGLangEmbeddingBench(
                device=device,
                data_type=data_type,
                model_path=model_path,
                embed_mode=embed_mode,
                attention_backend="intel_amx",
                enable_torch_compile=True,
            )
        else:
            print("no supported device specified, exit.")
            return
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

        # 一次迭代算 batch_size 个样本
        total_images = parallelism * batch_size * num_iter
        tps = total_images / total if total > 0 else 0.0

        print(
            f"total time: {total:.4f}s, "
            f"total_images={total_images}, TPS={tps:.2f} samples/s"
        )


def benchmark_embedding(
    parallelism: int,
    batch_size: int,
    num_iter: int,
    device: str,
    data_type: str,
    model_name: str,
    embed_mode: str,
):
    asyncio.run(
        benchmark_embedding_async(
            parallelism=parallelism,
            batch_size=batch_size,
            num_iter=num_iter,
            device=device,
            data_type=data_type,
            model_name=model_name,
            embed_mode=embed_mode,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SGLang offline embedding benchmark (text / text+image, random inputs)"
    )
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=100)

    # 两种用法：和原来一致
    parser.add_argument(
        "--num_iter",
        type=int,
        default=5,
        help="迭代次数（如果未提供 --total_images，则使用这个）",
    )
    parser.add_argument(
        "--total_images",
        type=int,
        default=None,
        help="总样本数（global），如果提供则覆盖 num_iter，"
             "要求 total_images 能被 parallelism * batch_size 整除",
    )

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--data_type",
        type=str,
        default="fp32",
        help="fp32 / fp16 / bf16，对应到 sglang 的 dtype 参数",
    )

    parser.add_argument(
        "--embed_mode",
        type=str,
        default="multimodal",
        choices=["text", "multimodal"],
        help="text: 只做文本 embedding；multimodal: 文本 + 图片 embedding",
    )

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

    # ====== 根据 total_images / num_iter 计算真实 num_iter ======
    if args.total_images is not None:
        per_step = args.parallelism * args.batch_size
        if per_step <= 0:
            raise ValueError("parallelism * batch_size 必须 > 0")

        if args.total_images % per_step != 0:
            raise ValueError(
                f"total_images={args.total_images} 不能被 "
                f"parallelism * batch_size = {per_step} 整除，"
                f"请调整 total_images 或 batch_size / parallelism。"
            )

        num_iter = args.total_images // per_step
        print(
            f"[INFO] 使用 total_images={args.total_images}, "
            f"parallelism={args.parallelism}, batch_size={args.batch_size} "
            f"得到 num_iter={num_iter}"
        )
    else:
        num_iter = args.num_iter
        print(
            f"[INFO] 未提供 total_images，使用 num_iter={num_iter}，"
            f"等效 total_images = parallelism * batch_size * num_iter = "
            f"{args.parallelism * args.batch_size * num_iter}"
        )

    benchmark_embedding(
        parallelism=args.parallelism,
        batch_size=args.batch_size,
        num_iter=num_iter,
        device=args.device,
        data_type=args.data_type,
        model_name=args.model,
        embed_mode=args.embed_mode,
    )
