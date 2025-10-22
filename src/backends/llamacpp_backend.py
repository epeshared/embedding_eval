
from typing import List
import torch

class LlamaCppEncoder:
    def __init__(self, model_path: str, n_threads: int=0, n_gpu_layers: int=0, verbose: bool=False):
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:
            raise RuntimeError(f"llama-cpp-python not installed or failed to import: {e}")
        print(f"[Init:llamacpp] model_path='{model_path}', n_threads={n_threads}, n_gpu_layers={n_gpu_layers}")
        self.llm = Llama(model_path=model_path, embedding=True,
                         n_threads=(n_threads or None), n_gpu_layers=n_gpu_layers, verbose=verbose)

    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int = 128) -> torch.Tensor:
        from numpy import array
        out = []
        for i in range(0, len(texts), batch_size):
            resp = self.llm.create_embedding(input=texts[i:i+batch_size])
            data = sorted(resp["data"], key=lambda d: d["index"])
            embs = torch.tensor(array([d["embedding"] for d in data]), dtype=torch.float32)
            out.append(torch.nn.functional.normalize(embs, p=2, dim=1))
        return torch.cat(out, dim=0) if out else torch.empty(0,0)
