
import os
from typing import Optional, List
import torch

class VLLMEncoder:
    def __init__(self, model: str, dtype: str="auto",
                 tensor_parallel_size: int=1, device: str="cuda",
                 max_model_len: Optional[int] = 8192,
                 gpu_memory_utilization: float = 0.90):
        try:
            from vllm import LLM  # type: ignore
        except Exception as e:
            raise RuntimeError(f"vllm not installed or import failed: {e}")

        print(f"[Init:vllm] model='{model}', dtype={dtype}, tp={tensor_parallel_size}, device={device}, "
              f"max_model_len={max_model_len}, gpu_mem_util={gpu_memory_utilization}")

        self._LLM = __import__("vllm", fromlist=["LLM"]).LLM
        llm_kwargs = dict(
            model=model,
            task="embed",
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            enforce_eager=False,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if max_model_len is not None and max_model_len > 0:
            llm_kwargs["max_model_len"] = max_model_len

        try:
            self.llm = self._LLM(**llm_kwargs)
        except TypeError:
            os.environ["VLLM_DEVICE"] = device
            self.llm = self._LLM(**llm_kwargs)

    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int = 128) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), batch_size):
            outputs = self.llm.embed(texts[i:i+batch_size])
            embs = torch.tensor([o.outputs.embedding for o in outputs])
            out.append(torch.nn.functional.normalize(embs, p=2, dim=1))
        return torch.cat(out, dim=0) if out else torch.empty(0,0)
