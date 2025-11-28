#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import argparse
import concurrent.futures
from typing import List, Optional

import numpy as np
from PIL import Image
import dataclasses

import sglang as sgl
from sglang.srt.server_args import ServerArgs
import torch
import io

# ====== 模型路径别名（按你的环境改） ======
MODEL_ALIASES = {
    # 小模型
    "clip-base": "openai/clip-vit-base-patch32",
    "openai/clip-vit-base-patch32": "openai/clip-vit-base-patch32",
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
        attention_backend: Optional[str] = None,
        enable_torch_compile: bool = False,
    ):
        self.text = "a beautiful landscape"
        # self.embed_mode = embed_mode

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
            # enable_trace="true",
        )

        engine_kwargs = dataclasses.asdict(server_args)
        print("[Init:sglang] Engine kwargs:", engine_kwargs)

        # ======== 纯 offline Engine（无 HTTP server） ========
        self.engine = sgl.Engine(**engine_kwargs)

        # ======== 根据模型名猜图像尺寸 ========
        if "336" in model_path or "large-patch14-336" in model_path:
            self.image_h = 336
            self.image_w = 336
        else:
            self.image_h = 224
            self.image_w = 224

        print(
            f"Init Finished! image_size=({self.image_h}, {self.image_w}), "
        )


    def run_once(
        self,
        embed_mode: str,
        texts: List[str],
        images: Optional[List] = None,
    ):
        if embed_mode == "text":
            return self.engine.encode(prompt=texts)

        elif embed_mode == "multimodal":
            assert images is not None

            image_data = []

            for i, (text_i, img_i) in enumerate(zip(texts, images)):
                # 统一成 PIL.Image 或 str
                if isinstance(img_i, Image.Image):
                    img_obj = img_i
                elif isinstance(img_i, np.ndarray):
                    img_obj = Image.fromarray(img_i)
                elif isinstance(img_i, str):
                    # 本地路径或 URL
                    img_obj = img_i
                else:
                    raise TypeError(f"Unsupported image type at index {i}: {type(img_i)}")

                image_data.append([img_obj])

            texts_with_token = ["<image> " + t for t in texts]

            out = self.engine.encode(
                prompt=texts_with_token,
                image_data=image_data,
            )
            return out

        else:
            raise ValueError(f"Unsupported embed_mode: {embed_mode}")




def run_benchmark_worker(
    instance: SGLangEmbeddingBench,
    batch_size: int,
    num_iter: int,
    embed_mode: str
):
    """单个 worker 线程：循环 num_iter 次调用 run_once。"""
    for _ in range(num_iter):
        if embed_mode == "text":
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

        t0 = time.time()
        instance.run_once(embed_mode, texts, images)
        print("current iter uses ", time.time() - t0, " s")


def benchmark_embedding(
    parallelism: int,
    batch_size: int,
    num_iter: int,
    device: str,
    data_type: str,
    model_name: str,
    embed_mode: str,
    profile: bool = False,
):
    model_path = resolve_model_path(model_name)
    print(
        f"开始并行测试: parallelism={parallelism}, batch_size={batch_size}, "
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
                attention_backend=None,      # 比如 "flashinfer"，按需再开
                enable_torch_compile=False,  # GPU 可以考虑 True，这里先关
            )
        elif device == "cpu":
            inst = SGLangEmbeddingBench(
                device=device,
                data_type=data_type,
                model_path=model_path,
                attention_backend="intel_amx",
                enable_torch_compile=True,
            )
        else:
            print("no supported device specified, exit.")
            return
        instances.append(inst)

    if profile:
        inst.engine.start_profile(record_shapes=True)

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = [
            executor.submit(run_benchmark_worker, inst, batch_size, num_iter, embed_mode)
            for inst in instances
        ]
        for f in futures:
            f.result()
    total = time.time() - start

    if profile:
        inst.engine.stop_profile()

    total_images = parallelism * batch_size * num_iter
    tps = total_images / total if total > 0 else 0.0

    print(
        f"total time: {total:.4f}s, "
        f"total_images={total_images}, TPS={tps:.2f} samples/s"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="SGLang offline embedding benchmark (text / text+image, random inputs)"
    )

    parser.add_argument("--validate", action="store_true", help="是否启用验证逻辑")

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

    parser.add_argument("--profile", action="store_true", help="是否启用性能分析")

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
        profile=args.profile,
    )