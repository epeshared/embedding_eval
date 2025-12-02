
from typing import List, Union, Optional
import os
import torch

# 额外依赖（用于 SGLang HTTP 调用）
try:
    import requests  # 用于 SGLang 原生 /encode 或 /v1/embeddings
    _REQUESTS_OK = True
except Exception:
    _REQUESTS_OK = False

def _to_float_list_list(data):
    return [list(map(float, row)) for row in data]

class SGLangEncoder:
    def __init__(self, base_url: str, model: str, api: str="native", api_key: str="", timeout: float=120.0):
        if not _REQUESTS_OK and api in ("native","v1"):
            raise RuntimeError("requests not installed (needed for SGLang native/v1 APIs)")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api = api.lower()
        self.api_key = api_key or ""
        self.timeout = timeout
        self.session = requests.Session() if _REQUESTS_OK else None
        print(f"[Init:sglang] url={self.base_url}, model='{self.model}', api='{self.api}'")

        self._openai_client = None
        if self.api == "openai":
            try:
                import openai  # type: ignore
            except Exception as e:
                raise RuntimeError(f"openai package not installed: {e}")
            self._openai_client = openai.Client(base_url=f"{self.base_url}/v1",
                                                api_key=(self.api_key or "None"))

    def start_profile(self, record_shapes: bool = False) -> bool:
        url = f"{self.base_url}/start_profile"
        payload = {"activities": ["CPU", "CUDA"]}
        if record_shapes:
            payload["record_shapes"] = True
        try:
            resp = self.session.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            print(f"[SGLang] /start_profile ok -> {resp.text}")
            return True
        except Exception as e:
            print(f"[SGLang] /start_profile failed: {e}")
            return False

    def stop_profile(self) -> bool:
        url = f"{self.base_url}/stop_profile"
        try:
            resp = self.session.post(url, json={}, timeout=self.timeout)
            resp.raise_for_status()
            print(f"[SGLang] /stop_profile ok -> {resp.text}")
            return True
        except Exception as e:
            print(f"[SGLang] /stop_profile failed: {e}")
            return False

    def _encode_native_one(self, text: str) -> List[float]:
        url = f"{self.base_url}/encode"
        payload = {"text": text}
        if self.model:
            payload["model"] = self.model
        resp = self.session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "embedding" in data:
            return list(map(float, data["embedding"]))
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list) and data["data"]:
            return list(map(float, data["data"][0]["embedding"]))
        raise RuntimeError(f"Unexpected response from /encode: {data}")

    def _encode_v1_any(self, inputs: Union[str, List[str]]) -> List[List[float]]:
        url = f"{self.base_url}/v1/embeddings"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model or "default",
            "input": inputs,
            "encoding_format": "bf16",
        }
        resp = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or "data" not in data or not isinstance(data["data"], list):
            raise RuntimeError(f"Unexpected response from /v1/embeddings: {data}")
        rows = sorted(data["data"], key=lambda d: d.get("index", 0))
        return [list(map(float, r["embedding"])) for r in rows]

    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int = 128) -> torch.Tensor:
        if not texts:
            return torch.empty(0, 0)
        out = []

        if self.api == "openai":
            if batch_size <= 1:
                for t in texts:
                    resp = self._openai_client.embeddings.create(model=(self.model or "default"), input=t)
                    rows = sorted(resp.data, key=lambda d: getattr(d, "index", 0))
                    vecs = [list(map(float, r.embedding)) for r in rows]
                    out.append(torch.nn.functional.normalize(torch.tensor(vecs, dtype=torch.float32), p=2, dim=1))
            else:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    resp = self._openai_client.embeddings.create(model=(self.model or "default"), input=batch)
                    rows = sorted(resp.data, key=lambda d: getattr(d, "index", 0))
                    vecs = [list(map(float, r.embedding)) for r in rows]
                    out.append(torch.nn.functional.normalize(torch.tensor(vecs, dtype=torch.float32), p=2, dim=1))
            return torch.cat(out, dim=0)

        if self.api == "v1":
            if batch_size <= 1:
                for t in texts:
                    vecs = self._encode_v1_any(t)
                    out.append(torch.nn.functional.normalize(torch.tensor(vecs, dtype=torch.float32), p=2, dim=1))
            else:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    vecs = self._encode_v1_any(batch)
                    out.append(torch.nn.functional.normalize(torch.tensor(vecs, dtype=torch.float32), p=2, dim=1))
            return torch.cat(out, dim=0)

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            vecs = []
            for s in batch:
                v = self._encode_native_one(s)
                vecs.append(v)
            t = torch.tensor(vecs, dtype=torch.float32)
            out.append(torch.nn.functional.normalize(t, p=2, dim=1))
        return torch.cat(out, dim=0) if out else torch.empty(0, 0)
