#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import argparse
import concurrent.futures
from typing import List, Optional

import numpy as np
from PIL import Image
from dataclasses import asdict

from vllm import LLM, EngineArgs

# ====== 默认模型根目录（可被 env / 命令行覆盖） ======
DEFAULT_MODEL_ROOT = "/home/xtang/embedding_eval/models"


def get_model_root(args_model_root: Optional[str]) -> str:
    """
    统一决定模型根目录的优先级：
      1) 命令行 --model_root
      2) 环境变量 MODEL_ROOT
      3) DEFAULT_MODEL_ROOT
    """
    return args_model_root or os.getenv("MODEL_ROOT") or DEFAULT_MODEL_ROOT


# 这里存的是“相对 MODEL_ROOT 的子路径”
MODEL_ALIASES = {
    # 小模型
    "clip-base": "openai/clip-vit-base-patch32",
    "openai/clip-vit-base-patch32": "openai/clip-vit-base-patch32",

    # 大模型（336）
    "clip-large-336": "openai/clip-vit-large-patch14-336",
    "openai/clip-vit-large-patch14-336": "openai/clip-vit-large-patch14-336",
}

DEFAULT_MODEL = "openai/clip-vit-base-patch32"


def resolve_model_path(model_name: str, model_root: str) -> str:
    """把命令行传进来的名字映射到本地目录或原样返回。"""
    if model_name in MODEL_ALIASES:
        rel = MODEL_ALIASES[model_name]
        return os.path.join(model_root, rel)
    return model_name


def map_data_type_to_vllm_dtype(data_type: str) -> str:
    """
    命令行 data_type -> vLLM EngineArgs.dtype 字符串。

    vLLM 通常支持:
      - "auto"
      - "float16"
      - "bfloat16"
      - "float32"
    """
    dt = data_type.lower()
    if dt in ["bf16", "bfloat16"]:
        return "bfloat16"
    if dt in ["fp16", "half", "float16"]:
        return "float16"
    if dt in ["fp32", "float32"]:
        return "float32"
    # 默认走 auto，让 vLLM 自己决定
    return "auto"


class VLLMEmbeddingBench:
    """
    使用 vLLM.LLM (offline) 做 embedding 基准测试（支持 text / text+image）。
    对齐原来 SGLangEmbeddingBench 的接口和用法。
    """

    def __init__(
        self,
        device: str,
        data_type: str,
        model_path: str,
        embed_mode: str = "multimodal",
        max_model_len: Optional[int] = None,
        tensor_parallel_size: int = 1,
    ):
        """
        embed_mode: "text" 或 "multimodal"
            - text:       只做文本 embedding
            - multimodal: 文本 + 图片 一起喂到 embed
        """
        # 和官方示例一致：多进程 worker 用 spawn
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        # 用 env 控制目标 device，而不是 EngineArgs.device
        dev = (device or "auto").lower()
        if dev in ("cuda", "cpu", "rocm", "xpu", "openvino"):
            os.environ["VLLM_TARGET_DEVICE"] = dev

        self.text = "a beautiful landscape"
        self.embed_mode = embed_mode

        # ======== dtype 处理 ========
        dtype_str = map_data_type_to_vllm_dtype(data_type)

        # ======== 限制多模态输入 ========
        # 为了省显存：默认 image=0；multimodal 时 image=1
        mm_limits = {"image": 0}
        if self.embed_mode == "multimodal":
            mm_limits["image"] = 1

        # ======== 组装 EngineArgs（有 max_model_len 才传，避免超过模型上限报错） ========
        ea_kwargs = dict(
            model=model_path,
            task="embed",  # 声明是 embedding 任务
            dtype=dtype_str,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt=mm_limits,
            trust_remote_code=True,
            disable_log_stats=True,
        )
        if max_model_len is not None:
            ea_kwargs["max_model_len"] = max_model_len

        engine_args = EngineArgs(**ea_kwargs)
        engine_kwargs = asdict(engine_args)
        print("[Init:vLLM] Engine kwargs:", engine_kwargs)

        # ======== 纯 offline LLM（无 HTTP server） ========
        self.llm = LLM(**engine_kwargs)

        # ======== 根据模型名猜图像尺寸（沿用你原来的逻辑） ========
        if "336" in model_path or "large-patch14-336" in model_path:
            self.image_h = 336
            self.image_w = 336
        else:
            self.image_h = 224
            self.image_w = 224

        print(
            f"Init Finished! image_size=({self.image_h}, {self.image_w}), "
            f"embed_mode={self.embed_mode}, "
            f"VLLM_TARGET_DEVICE={os.environ.get('VLLM_TARGET_DEVICE', 'auto')}"
        )

    def run_once(self, texts: List[str], images: Optional[List[np.ndarray]] = None):
        """用给定的 texts / images 跑一次 embedding。"""
        if self.embed_mode == "text":
            # 纯文本 embedding：直接传 list[str]
            outputs = self.llm.embed(texts)
            _ = outputs
        elif self.embed_mode == "multimodal":
            assert images is not None, "multimodal 模式下必须提供 images"

            # 把 numpy 转成 PIL.Image
            pil_images = [Image.fromarray(img) for img in images]

            # vLLM 多模态 embedding 输入格式：
            # 每条样本一个 dict:
            # {
            #   "prompt": <text>,
            #   "multi_modal_data": {"image": <PIL.Image or np.ndarray>}
            # }
            prompts = []
            for t, im in zip(texts, pil_images):
                prompts.append(
                    {
                        "prompt": t,
                        "multi_modal_data": {"image": im},
                    }
                )

            outputs = self.llm.embed(prompts)
            _ = outputs
        else:
            raise ValueError(f"Unsupported embed_mode: {self.embed_mode}")


def run_benchmark_worker(
    instance: VLLMEmbeddingBench,
    batch_size: int,
    num_iter: int,
):
    """单个 worker 线程：循环 num_iter 次调用 run_once。"""
    for _ in range(num_iter):
        if instance.embed_mode == "text":
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


def benchmark_embedding(
    parallelism: int,
    batch_size: int,
    num_iter: int,
    device: str,
    data_type: str,
    model_name: str,
    embed_mode: str,
    model_root: str,
):
    model_path = resolve_model_path(model_name, model_root)
    print(
        f"开始并行测试: parallelism={parallelism}, batch_size={batch_size}, "
        f"num_iter={num_iter}, device={device}, data_type={data_type}, "
        f"embed_mode={embed_mode}, model={model_path}"
    )

    instances: List[VLLMEmbeddingBench] = []
    for _ in range(parallelism):
        inst = VLLMEmbeddingBench(
            device=device,
            data_type=data_type,
            model_path=model_path,
            embed_mode=embed_mode,
            max_model_len=None,       # 让 vLLM 自己从 config 推导，避免超过上限
            tensor_parallel_size=1,
        )
        instances.append(inst)

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = [
            executor.submit(run_benchmark_worker, inst, batch_size, num_iter)
            for inst in instances
        ]
        for f in futures:
            f.result()
    total = time.time() - start

    total_samples = parallelism * batch_size * num_iter
    tps = total_samples / total if total > 0 else 0.0

    print(
        f"total time: {total:.4f}s, "
        f"total_samples={total_samples}, TPS={tps:.2f} samples/s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="vLLM offline vision-language embedding benchmark "
                    "(text / text+image, random inputs)"
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

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="vLLM 目标设备: auto / cuda / cpu / xpu / rocm / openvino",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="fp16",
        help="fp32 / fp16 / bf16，对应到 vLLM 的 dtype 参数",
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
            "当前别名沿用原来的 CLIP；实际使用时建议换成 vLLM 支持的 VLM embedding 模型，"
            "比如 TIGER-Lab/VLM2Vec-Full 等。"
        ),
    )

    parser.add_argument(
        "--model_root",
        type=str,
        default=None,
        help=(
            "模型根目录，可覆盖默认值和环境变量 MODEL_ROOT。"
            "优先级: --model_root > $MODEL_ROOT > /home/xtang/models"
        ),
    )

    args = parser.parse_args()

    model_root = get_model_root(args.model_root)
    print(f"[INFO] 使用模型根目录: {model_root}")

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
        model_root=model_root,
    )
