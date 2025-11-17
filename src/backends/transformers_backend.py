
import os
from typing import Optional, List
import torch
from ..utils import mean_pooling, l2_normalize
try:
    from transformers import AutoModel, AutoTokenizer
    _TRANSFORMERS_OK = True
except Exception:
    _TRANSFORMERS_OK = False

def _robust_load_tok_model(model_id_or_path: str, trust_remote_code: bool, offline: bool):
    is_local = os.path.isdir(model_id_or_path)
    local_only = offline or is_local
    def _do_load():
        tok = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code, local_files_only=local_only)
        mdl = AutoModel.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code, local_files_only=local_only)
        return tok, mdl
    if local_only:
        return _do_load()
    last_err = None
    endpoint_before = os.environ.get("HF_ENDPOINT")
    try:
        return _do_load()
    except Exception as e:
        last_err = e
        if endpoint_before:
            os.environ.pop("HF_ENDPOINT", None)
    try:
        return _do_load()
    except Exception as e:
        if endpoint_before:
            os.environ["HF_ENDPOINT"] = endpoint_before
        raise last_err if last_err is not None else e

class TransformersEncoder:
    def __init__(self, model_name: str, device: str="cpu", use_ipex: str="True",
                 amp: str="auto", max_length: int=512, offline: bool=False,
                 trust_remote_code: bool=True):
        if not _TRANSFORMERS_OK:
            raise RuntimeError("transformers not installed")
        self.device = torch.device(device)
        self.use_ipex = (str(use_ipex).lower() == "true")
        self.amp = amp.lower()
        self.max_length = max_length
        print(f"[Init:transformers] model='{model_name}', device={self.device}, ipex={self.use_ipex}, amp={self.amp}, offline={offline}")
        self.tokenizer, self.model = _robust_load_tok_model(model_name, trust_remote_code, offline)
        self.model = self.model.to(self.device).eval()

        self.cpu_amp_dtype: Optional[torch.dtype] = None
        self._ipex_enabled = False
        if self.device.type == "cpu" and self.use_ipex:
            try:
                import intel_extension_for_pytorch as ipex  # noqa: F401
                self.model = ipex.optimize(self.model, dtype=torch.bfloat16, inplace=True)
                self.cpu_amp_dtype = torch.bfloat16
                self._ipex_enabled = True
                print("[Init:transformers] IPEX enabled (bf16).")
            except Exception as e:
                print(f"[Warn] IPEX import/optimize failed, fallback to plain PyTorch: {e}")

    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int = 128) -> torch.Tensor:
        out = []
        dev = self.device.type
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            if dev == "cuda":
                if self.amp == "off":
                    hs = self.model(**enc).last_hidden_state
                elif self.amp == "fp16":
                    with torch.autocast("cuda", dtype=torch.float16):
                        hs = self.model(**enc).last_hidden_state
                elif self.amp == "bf16":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        hs = self.model(**enc).last_hidden_state
                else:
                    with torch.autocast("cuda"):
                        hs = self.model(**enc).last_hidden_state
            elif dev == "cpu" and self._ipex_enabled and self.cpu_amp_dtype is not None and self.amp in ("bf16","auto"):
                with torch.autocast("cpu", dtype=self.cpu_amp_dtype):
                    hs = self.model(**enc).last_hidden_state
            else:
                hs = self.model(**enc).last_hidden_state
            emb = l2_normalize(mean_pooling(hs, enc["attention_mask"]), dim=1)
            out.append(emb.cpu())
        h = getattr(self.model.config,"hidden_size",1024)
        return torch.cat(out, dim=0) if out else torch.empty(0, h)
