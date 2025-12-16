import os
from typing import Optional, List, Any, Union
import torch
import dataclasses

from sglang.srt.server_args import ServerArgs  # type: ignore[import-untyped]
import sglang as sgl  # type: ignore[import-untyped]


class SGLangOfflineEncoder:
    """Offline embedding encoder using sglang.Engine + ServerArgs."""

    def __init__(
        self,
        model: str,
        dtype: str = "auto",
        device: str = "cuda",
        tp_size: int = 1,
        dp_size: int = 1,
        random_seed: int = 0,
        trust_remote_code: bool = False,
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        attention_backend: Optional[str] = None,
        is_embedding: bool = True,
        enable_torch_compile: bool = True,
        torch_compile_max_bs: int = 32,
        **engine_extra_kwargs,
    ):
        # ==== Step 1: 组装 ServerArgs ====
        # CLI 参数本来通过 add_cli_args() + parse_args() 来获得
        server_args = ServerArgs(
            model_path=model,
            dtype=dtype,
            device=device,
            tp_size=tp_size,
            dp_size=dp_size,
            random_seed=random_seed,
            trust_remote_code=trust_remote_code,
            quantization=quantization,
            revision=revision,
            is_embedding=is_embedding,
            enable_torch_compile=enable_torch_compile,
            torch_compile_max_bs=16,
            attention_backend=attention_backend,
            log_level="error",
            **engine_extra_kwargs,         # <==== 保留扩展参数的通道
        )

        # ==== Step 2: 转为 dict 传入 Engine ====
        engine_kwargs = dataclasses.asdict(server_args)

        print("[Init:sglang] Engine kwargs:", engine_kwargs)

        # ======= 纯 offline 模式 ========
        self.engine = sgl.Engine(**engine_kwargs)  # type: ignore[attr-defined]

    @torch.inference_mode()
    def encode(
        self,
        texts: List[str],
        batch_size: int = 128,
        normalize: bool = True,
    ) -> torch.Tensor:
        if not texts:
            return torch.empty(0, 0)

        all_chunks = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            outputs = self.engine.encode(batch)
            embs = torch.tensor([o["embedding"] for o in outputs])
            if normalize:
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_chunks.append(embs)

        return torch.cat(all_chunks, dim=0)

    @torch.inference_mode()
    def encode_images(
        self,
        images: List[Union[Any, str, dict]],
        batch_size: int = 128,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Encode images into embeddings.

        Args:
            images: One image per sample. Each item can be a PIL.Image.Image,
                a local file path, an URL, a base64 string, or an sglang ImageData-like dict.
                (See sglang's multimodal input conventions.)
            batch_size: Number of images per batch.
            normalize: Whether to L2-normalize embeddings.

        Returns:
            A tensor of shape (N, D).
        """
        if not images:
            return torch.empty(0, 0)

        all_chunks: List[torch.Tensor] = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            # For image-only embedding, provide a dummy prompt; sglang requires `prompt`.
            dummy_prompts = [""] * len(batch_images)
            outputs = self.engine.encode(dummy_prompts, image_data=batch_images)

            # sglang returns a list of dicts: [{"embedding": [...], "meta_info": {...}}, ...]
            embs = torch.tensor([o["embedding"] for o in outputs])
            if normalize:
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_chunks.append(embs)

        return torch.cat(all_chunks, dim=0)


# if __name__ == "__main__":
#     encoder = SGLangOfflineEncoder(
#         model="/home/xtang/embedding_eval/models/Qwen/Qwen3-Embedding-4B",
#         dtype="auto",
#         device="cpu",
#         tp_size=1,
#         dp_size=1,
#         is_embedding=True,
#         enable_torch_compile=True,
#         torch_compile_max_bs=16,
#         attention_backend=None,
#     )

#     texts = ["hello world", "sglang offline engine test"]
#     embs = encoder.encode(texts, batch_size=2)
#     print("embeddings shape:", embs.shape)
#     print(embs)
