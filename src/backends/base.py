
from typing import List
import torch

class BaseEncoder:
    def encode(self, texts: List[str], batch_size: int = 128) -> torch.Tensor:
        raise NotImplementedError
