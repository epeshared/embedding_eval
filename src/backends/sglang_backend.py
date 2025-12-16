
from typing import List, Union, Optional, Any, Dict
import os
import torch
import base64
import mimetypes

# 额外依赖（用于 SGLang HTTP 调用）
try:
    import requests  # type: ignore  # 用于 SGLang 原生 /encode 或 /v1/embeddings
    _REQUESTS_OK = True
except Exception:
    _REQUESTS_OK = False

def _to_float_list_list(data):
    return [list(map(float, row)) for row in data]

class SGLangEncoder:
    def __init__(
        self,
        base_url: str,
        model: str,
        api: str = "native",
        api_key: str = "",
        timeout: float = 120.0,
        image_transport: str = "data-url",
    ):
        if not _REQUESTS_OK and api in ("native", "v1", "openai"):
            raise RuntimeError("requests not installed (needed for SGLang native/v1/openai HTTP APIs)")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api = api.lower()
        self.api_key = api_key or ""
        self.timeout = timeout
        self.image_transport = (image_transport or "data-url").lower()
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
        session = self.session
        assert session is not None
        payload: Dict[str, Any] = {
            "activities": ["CPU", "CUDA"],
            **({"record_shapes": True} if record_shapes else {}),
        }
        try:
            resp = session.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            print(f"[SGLang] /start_profile ok -> {resp.text}")
            return True
        except Exception as e:
            print(f"[SGLang] /start_profile failed: {e}")
            return False

    def stop_profile(self) -> bool:
        url = f"{self.base_url}/stop_profile"
        session = self.session
        assert session is not None
        try:
            resp = session.post(url, json={}, timeout=self.timeout)
            resp.raise_for_status()
            print(f"[SGLang] /stop_profile ok -> {resp.text}")
            return True
        except Exception as e:
            print(f"[SGLang] /stop_profile failed: {e}")
            return False

    def _encode_native_one(self, text: str) -> List[float]:
        url = f"{self.base_url}/encode"
        session = self.session
        assert session is not None
        payload = {"text": text}
        if self.model:
            payload["model"] = self.model
        resp = session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "embedding" in data:
            return list(map(float, data["embedding"]))
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list) and data["data"]:
            return list(map(float, data["data"][0]["embedding"]))
        raise RuntimeError(f"Unexpected response from /encode: {data}")

    def _encode_v1_any(self, inputs: Union[str, List[str]]) -> List[List[float]]:
        url = f"{self.base_url}/v1/embeddings"
        session = self.session
        assert session is not None
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model or "default",
            "input": inputs,
            "encoding_format": "bf16",
        }
        resp = session.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or "data" not in data or not isinstance(data["data"], list):
            raise RuntimeError(f"Unexpected response from /v1/embeddings: {data}")
        rows = sorted(data["data"], key=lambda d: d.get("index", 0))
        return [list(map(float, r["embedding"])) for r in rows]

    def _encode_v1_multimodal_any(self, inputs: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[List[float]]:
        """Call /v1/embeddings with multimodal items (e.g., {"text": ...} or {"image": ...})."""
        url = f"{self.base_url}/v1/embeddings"
        session = self.session
        assert session is not None
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: Dict[str, Any] = {
            "model": self.model or "default",
            "input": inputs,
        }
        resp = session.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or "data" not in data or not isinstance(data["data"], list):
            raise RuntimeError(f"Unexpected response from /v1/embeddings: {data}")
        rows = sorted(data["data"], key=lambda d: d.get("index", 0))
        return [list(map(float, r["embedding"])) for r in rows]

    def _image_to_repr(self, img: Any) -> Any:
        """Normalize an image input into a server-understandable representation.

        - If image_transport is 'path/url', pass through strings.
        - If image_transport is 'data-url' and img is a local file path, convert to data URL.
        - If img is already a data URL / URL / base64-ish string, pass through.
        """
        if self.image_transport == "path/url":
            return img

        # data-url
        if isinstance(img, str):
            s = img.strip()
            if s.startswith("data:") or s.startswith("http://") or s.startswith("https://"):
                return s
            if os.path.exists(s) and os.path.isfile(s):
                mime, _ = mimetypes.guess_type(s)
                mime = mime or "application/octet-stream"
                with open(s, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                return f"data:{mime};base64,{b64}"
            # assume already acceptable string
            return s

        # bytes -> data-url (unknown mime)
        if isinstance(img, (bytes, bytearray)):
            b64 = base64.b64encode(bytes(img)).decode("ascii")
            return f"data:application/octet-stream;base64,{b64}"

        # dict / PIL.Image / others: let server-side handle if supported
        return img

    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int = 128) -> torch.Tensor:
        if not texts:
            return torch.empty(0, 0)
        out: List[torch.Tensor] = []

        if self.api == "openai":
            client = self._openai_client
            assert client is not None
            if batch_size <= 1:
                for text in texts:
                    resp = client.embeddings.create(model=(self.model or "default"), input=text)
                    rows = sorted(resp.data, key=lambda d: getattr(d, "index", 0))
                    vecs = [list(map(float, r.embedding)) for r in rows]
                    out.append(torch.nn.functional.normalize(torch.tensor(vecs, dtype=torch.float32), p=2, dim=1))
            else:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    resp = client.embeddings.create(model=(self.model or "default"), input=batch)
                    rows = sorted(resp.data, key=lambda d: getattr(d, "index", 0))
                    vecs = [list(map(float, r.embedding)) for r in rows]
                    out.append(torch.nn.functional.normalize(torch.tensor(vecs, dtype=torch.float32), p=2, dim=1))
            return torch.cat(out, dim=0)

        if self.api == "v1":
            if batch_size <= 1:
                for text in texts:
                    vecs = self._encode_v1_any(text)
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
            emb_t = torch.tensor(vecs, dtype=torch.float32)
            out.append(torch.nn.functional.normalize(emb_t, p=2, dim=1))
        return torch.cat(out, dim=0) if out else torch.empty(0, 0)

    @torch.inference_mode()
    def encode_images(self, images: List[Any], batch_size: int = 128) -> torch.Tensor:
        """Image embedding via SGLang server.

        Supported APIs:
          - v1: uses /v1/embeddings with multimodal items {"image": ...}
          - openai: uses the same HTTP /v1/embeddings (openai python client may not accept dict inputs)

        Notes:
          - For remote servers without shared filesystem, prefer image_transport='data-url'.
          - For shared filesystem, image_transport='path/url' can reduce client overhead.
        """
        if not images:
            return torch.empty(0, 0)

        if self.api not in ("v1", "openai"):
            raise RuntimeError(
                "Image embeddings are supported only for api='v1' or api='openai' (HTTP /v1/embeddings)."
            )

        session = self.session
        assert session is not None

        out: List[torch.Tensor] = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = [{"image": self._image_to_repr(x)} for x in batch]
            vecs = self._encode_v1_multimodal_any(inputs)
            t = torch.tensor(vecs, dtype=torch.float32)
            out.append(torch.nn.functional.normalize(t, p=2, dim=1))

        return torch.cat(out, dim=0) if out else torch.empty(0, 0)
