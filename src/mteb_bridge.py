
import os
from typing import List, Optional
import torch

def _to_numpy(t: torch.Tensor):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t

class MTEBModelAdapter:
    def __init__(self, encoder, batch_size: int = 128):
        self.encoder = encoder
        self.batch_size = batch_size

    def encode(self, sentences, **kwargs):
        texts = list(sentences)
        embs = self.encoder.encode(texts, batch_size=self.batch_size)
        return _to_numpy(embs)

    def encode_queries(self, queries, **kwargs):
        return self.encode(queries, **kwargs)

    def encode_documents(self, docs, **kwargs):
        if len(docs) > 0 and isinstance(docs[0], dict):
            texts = [d.get("text", d.get("title", "")) for d in docs]
        else:
            texts = list(docs)
        return self.encode(texts, **kwargs)

@torch.inference_mode()
def run_mteb(encoder,
             tasks: List[str],
             output_dir: str,
             batch_size: int = 128,
             task_langs: Optional[List[str]] = None,
             task_types: Optional[List[str]] = None):
    try:
        from mteb import MTEB
    except Exception as e:
        raise RuntimeError(
            "未安装 mteb 库。请先安装：\n  pip install mteb"
        ) from e

    os.makedirs(output_dir, exist_ok=True)
    model = MTEBModelAdapter(encoder, batch_size=batch_size)

    if tasks:
        evaluation = MTEB(tasks=tasks, task_langs=task_langs, task_types=task_types)
    else:
        evaluation = MTEB(task_langs=task_langs, task_types=task_types)

    print(f"[MTEB] tasks={tasks or '<auto by filters>'} langs={task_langs} types={task_types} -> out={output_dir}")
    evaluation.run(model, output_folder=output_dir)
    print("[MTEB] Done.")
