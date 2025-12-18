import os
from typing import List, Optional, Union, Any
import torch

class CLIPEncoder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                device: str = "cpu", use_ipex: str = "True",
                amp: str = "auto", offline: bool = False):
        from transformers import CLIPModel, AutoProcessor
        self.device = torch.device(device)
        self.use_ipex = (str(use_ipex).lower() == "true")
        self.amp = amp.lower()
        self.offline = (str(offline).lower() == "true")

        is_local = os.path.isdir(model_name)
        local_only = self.offline or is_local

        print(f"[Init:CLIP] model='{model_name}', device={self.device}, ipex={self.use_ipex}, "
              f"amp={self.amp}, offline={self.offline}")

        self.processor = AutoProcessor.from_pretrained(
            model_name, local_files_only=local_only
        )
        self.model = CLIPModel.from_pretrained(
            model_name, local_files_only=local_only
        ).to(self.device).eval()

        self.cpu_amp_dtype: Optional[torch.dtype] = None
        self._IPEX_enabled = False
        if self.device.type == "cpu" and self.use_ipex:
            try:
                import importlib

                ipex = importlib.import_module("intel_extension_for_pytorch")
                self.model = ipex.optimize(self.model, dtype=torch.bfloat16, inplace=True)
                self.cpu_amp_dtype = torch.bfloat16
                self._IPEX_enabled = True
                print("[Init:CLIP] IPEX enabled (bf16).")
            except Exception as e:
                print(f"[Warn] IPEX optimize failed, fallback to plain PyTorch: {e}")
                self.cpu_amp_dtype = torch.bfloat16
                self.model.to(torch.bfloat16)

    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int = 256, normalize: bool = True) -> torch.Tensor:
        feats = self.encode_text(texts, batch_size=batch_size)
        if normalize:
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats

    @torch.inference_mode()
    def encode_text(self, prompts: List[str], batch_size: int = 256) -> torch.Tensor:
        dev = self.device.type
        feats = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            if dev == "cpu" and getattr(self, "_IPEX_enabled", False) and self.cpu_amp_dtype is not None and self.amp in ("bf16","auto"):
                with torch.autocast("cpu", dtype=self.cpu_amp_dtype):
                    t = self.model.get_text_features(**inputs)
            elif dev == "cuda" and self.amp != "off":
                dtype = torch.float16 if self.amp == "fp16" else (torch.bfloat16 if self.amp == "bf16" else None)
                ctx = torch.autocast("cuda", dtype=dtype) if dtype else torch.autocast("cuda")
                with ctx:
                    t = self.model.get_text_features(**inputs)
            else:
                t = self.model.get_text_features(**inputs)
            feats.append(torch.nn.functional.normalize(t, p=2, dim=-1).cpu())
        return torch.cat(feats, dim=0)

    @torch.inference_mode()
    def encode_images(self, images: List[Union[str, Any]], batch_size: int = 128) -> torch.Tensor:
        dev = self.device.type
        feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]

            # Flickr8k pipeline passes file paths; support both PIL images and local paths.
            if batch and isinstance(batch[0], str):
                from PIL import Image

                pil_batch = []
                for p in batch:
                    if not isinstance(p, str):
                        raise TypeError(f"encode_images received mixed types in batch: {type(p)}")
                    if not os.path.exists(p):
                        raise FileNotFoundError(f"Image path not found: {p}")
                    with Image.open(p) as im:
                        pil_batch.append(im.convert("RGB"))
                batch_inputs = pil_batch
            else:
                batch_inputs = batch

            inputs = self.processor(images=batch_inputs, return_tensors="pt").to(self.device)
            if dev == "cpu" and getattr(self, "_IPEX_enabled", False) and self.cpu_amp_dtype is not None and self.amp in ("bf16","auto"):
                with torch.autocast("cpu", dtype=self.cpu_amp_dtype):
                    v = self.model.get_image_features(**inputs)
            elif dev == "cuda" and self.amp != "off":
                dtype = torch.float16 if self.amp == "fp16" else (torch.bfloat16 if self.amp == "bf16" else None)
                ctx = torch.autocast("cuda", dtype=dtype) if dtype else torch.autocast("cuda")
                with ctx:
                    v = self.model.get_image_features(**inputs)
            else:
                v = self.model.get_image_features(**inputs)
            feats.append(torch.nn.functional.normalize(v, p=2, dim=-1).cpu())
        return torch.cat(feats, dim=0)
