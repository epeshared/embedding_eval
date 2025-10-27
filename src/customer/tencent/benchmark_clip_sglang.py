#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import io
import os
import base64
import time
from typing import List, Union, Optional, Tuple, Dict

import torch
import numpy as np
from PIL import Image
import requests


# =========================
# 工具函数
# =========================
def np_to_png_bytes(arr: np.ndarray) -> bytes:
    """HWC uint8 (RGB) -> PNG bytes"""
    # Pillow 现在不建议传 mode 参数，直接让它推断
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def png_bytes_to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def read_flickr8k_captions(captions_file: str) -> Dict[str, List[str]]:
    """Flickr8k.token.txt -> {filename: [captions...]}"""
    mp: Dict[str, List[str]] = {}
    with open(captions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            left, cap = line.split("\t", 1)
            img, _ = left.split("#", 1)
            mp.setdefault(img, []).append(cap)
    return mp


def load_flickr8k_pairs(
    images_dir: str,
    captions_file: str,
    total: int,
    pick_caption: str = "first",
) -> Tuple[List[np.ndarray], List[str]]:
    """返回 total 对 (image np.uint8 HWC, text)"""
    cap_map = read_flickr8k_captions(captions_file)
    filenames = [
        fn for fn in os.listdir(images_dir)
        if fn.lower().endswith((".jpg", ".jpeg", ".png")) and fn in cap_map
    ]
    if not filenames:
        raise RuntimeError(f"{images_dir} 中没有与 {os.path.basename(captions_file)} 匹配的图片。")
    filenames.sort()
    rng = np.random.default_rng()

    images, texts = [], []
    n = len(filenames)
    for i in range(total):
        fn = filenames[i % n]
        with Image.open(os.path.join(images_dir, fn)) as im:
            im = im.convert("RGB")
            arr = np.array(im, dtype=np.uint8)
        caps = cap_map[fn]
        txt = caps[rng.integers(0, len(caps))] if pick_caption == "random" else caps[0]
        images.append(arr)
        texts.append(txt)
    return images, texts


def make_random_images_and_texts(total: int) -> Tuple[List[np.ndarray], List[str]]:
    """与你原逻辑一致：随机 224x224 图 + 相同文本"""
    images = [
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        for _ in range(total)
    ]
    texts = ["a beautiful landscape"] * total
    return images, texts


def make_random_texts(total: int) -> List[str]:
    """简单：全部相同文本"""
    return ["a beautiful landscape"] * total


# =========================
# SGLangEncoder
# =========================
class SGLangEncoder:
    """
    统一封装 SGLang 三种 API：
      - api="native": POST /encode （单条文本）
      - api="v1"    : POST /v1/embeddings （OpenAI 兼容，多条）
      - api="openai": openai.Client(...).embeddings.create(...)
    默认用 v1。
    """

    def __init__(self, base_url: str, model: str,
                 api: str = "v1", api_key: str = "", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api = api.lower()
        self.api_key = api_key or ""
        self.timeout = timeout
        self.session = requests.Session()

        print(f"[Init:sglang] url={self.base_url}, model='{self.model}', api='{self.api}'")

        self._openai_client = None
        if self.api == "openai":
            try:
                import openai  # type: ignore
            except Exception as e:
                raise RuntimeError(f"openai package not installed: {e}")
            self._openai_client = openai.Client(
                base_url=f"{self.base_url}/v1",
                api_key=(self.api_key or "None")
            )

    # ---------- native ----------
    def _encode_native_one(self, text: str) -> List[float]:
        url = f"{self.base_url}/encode"
        payload = {"text": text, "model": self.model}
        resp = self.session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "embedding" in data:
            return [float(x) for x in data["embedding"]]
        if isinstance(data, dict) and "data" in data and data["data"]:
            return [float(x) for x in data["data"][0]["embedding"]]
        raise RuntimeError(f"Unexpected /encode response: {data}")

    # ---------- v1 (文本) ----------
    def _encode_v1_any(self, inputs: Union[str, List[str]]) -> List[List[float]]:
        url = f"{self.base_url}/v1/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model or "default",
            "input": inputs,            # str 或 List[str]
            "encoding_format": "float", # 如需 'bf16' 可改
        }
        resp = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"/v1/embeddings text {resp.status_code}: {resp.text[:800]}")
        js = resp.json()
        if not isinstance(js, dict) or "data" not in js or not isinstance(js["data"], list):
            raise RuntimeError(f"Unexpected /v1/embeddings response: {js}")
        rows = sorted(js["data"], key=lambda d: d.get("index", 0))
        vecs = []
        for r in rows:
            emb = r.get("embedding", r.get("vector", None))
            if emb is None:
                raise RuntimeError(f"Missing 'embedding' in item: {r}")
            vecs.append([float(x) for x in emb])
        return vecs

    # ---------- 统一 encode（文本） ----------
    @torch.inference_mode()
    def encode_texts(self, texts: List[str], batch_size: int = 128) -> torch.Tensor:
        if not texts:
            return torch.empty(0, 0)
        out = []
        if self.api == "openai":
            resp_join = []
            if batch_size <= 1:
                for t in texts:
                    resp = self._openai_client.embeddings.create(model=(self.model or "default"), input=t)
                    resp_join.append(resp)
            else:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    resp = self._openai_client.embeddings.create(model=(self.model or "default"), input=batch)
                    resp_join.append(resp)
            for resp in resp_join:
                rows = sorted(resp.data, key=lambda d: getattr(d, "index", 0))
                vecs = [[float(x) for x in r.embedding] for r in rows]
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

        # native
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            vecs = [self._encode_native_one(s) for s in batch]
            t = torch.tensor(vecs, dtype=torch.float32)
            out.append(torch.nn.functional.normalize(t, p=2, dim=1))
        return torch.cat(out, dim=0) if out else torch.empty(0, 0)

    # ---------- v1 多模态：文本+图片（pairlist 版） ----------
    def encode_v1_multimodal_pairlist(
        self,
        texts: List[str],
        images: List[Union[str, bytes, np.ndarray]],
        batch_size: int = 32,
        image_transport: str = "data-url",  # "data-url" | "path/url"
    ) -> torch.Tensor:
        """
        多模态（文本+图片） -> /v1/embeddings
        使用 pairlist 负载（官网简化版的批量形式）：
        {
          "model": "...",
          "input": [ {"text": "...", "image": "<data-url 或 url>"}, {...}, ... ],
          "encoding_format": "float"
        }
        """
        assert len(texts) == len(images), "texts 与 images 数量需一致"
        if self.api != "v1":
            raise RuntimeError("多模态当前仅通过 /v1/embeddings 实现，请设置 api='v1'")
        if not texts:
            return torch.empty(0, 0)

        url = f"{self.base_url}/v1/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        outs = []
        for i in range(0, len(texts), batch_size):
            sub_t = texts[i:i+batch_size]
            sub_i = images[i:i+batch_size]

            # 构造 pairlist：每个样本是 {"text": "...", "image": "..."}
            pairlist = []
            for t, img in zip(sub_t, sub_i):
                if image_transport == "data-url":
                    if isinstance(img, bytes):
                        img_repr = png_bytes_to_data_url(img)
                    elif isinstance(img, np.ndarray):
                        img_repr = png_bytes_to_data_url(np_to_png_bytes(img))
                    elif isinstance(img, str) and img.startswith("data:image/"):
                        img_repr = img  # 已是 data-url
                    else:
                        raise ValueError("image=data-url 需要 numpy.ndarray / PNG bytes / 已是 data-url 的字符串")
                else:
                    # path/url：传可访问的 URL（或后端可见路径）
                    if not isinstance(img, str) or not img:
                        raise ValueError("image_transport='path/url' 需要字符串路径或 URL")
                    img_repr = img

                pairlist.append({"text": t, "image": img_repr})

            payload = {
                "model": self.model or "default",
                "input": pairlist,
                "encoding_format": "float",
            }

            resp = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"/v1/embeddings multimodal {resp.status_code}: {resp.text[:1000]}")
            js = resp.json()
            if not isinstance(js, dict) or "data" not in js or not isinstance(js["data"], list):
                raise RuntimeError(f"Unexpected /v1/embeddings (mm) response: {js}")

            rows = sorted(js["data"], key=lambda d: d.get("index", 0))
            vecs = []
            for r in rows:
                emb = r.get("embedding", r.get("vector", None))
                if emb is None:
                    raise RuntimeError(f"Missing 'embedding' in item: {r}")
                vecs.append([float(x) for x in emb])
            t = torch.tensor(vecs, dtype=torch.float32)
            outs.append(torch.nn.functional.normalize(t, p=2, dim=1))

        return torch.cat(outs, dim=0) if outs else torch.empty(0, 0)


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser("SGLang embeddings | 文本 & 文本+图片(pairlist) | 随机 / Flickr8k")
    # 服务
    ap.add_argument("--base_url", type=str, default="http://127.0.0.1:30000")
    ap.add_argument("--model", type=str, default="gme-qwen2-vl")
    ap.add_argument("--api", choices=["v1", "native", "openai"], default="v1")
    ap.add_argument("--api_key", type=str, default="")
    ap.add_argument("--timeout", type=float, default=120.0)

    # 模式
    ap.add_argument("--mode", choices=["text", "multimodal"], default="text",
                    help="text=纯文本；multimodal=文本+图片（pairlist payload）")
    ap.add_argument("--data_source", choices=["random", "flickr8k"], default="random")

    # Flickr8k
    ap.add_argument("--flickr_images_dir", type=str, default=None)
    ap.add_argument("--flickr_captions_file", type=str, default=None)
    ap.add_argument("--flickr_caption_pick", choices=["first", "random"], default="first")

    # 负载大小
    ap.add_argument("--num_samples", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)

    # 多模态图片传输
    ap.add_argument("--image_transport", choices=["data-url", "path/url"], default="data-url",
                    help="data-url 走 base64 PNG；path/url 直接路径或 URL（需后端可访问）")

    args = ap.parse_args()

    enc = SGLangEncoder(
        base_url=args.base_url,
        model=args.model,
        api=args.api,
        api_key=args.api_key,
        timeout=args.timeout,
    )

    t0 = time.time()
    if args.mode == "text":
        # 准备文本
        if args.data_source == "random":
            texts = make_random_texts(args.num_samples)
        else:
            # 用 Flickr8k 的 caption 做文本（不读图）
            if not args.flickr_captions_file:
                raise ValueError("--data_source flickr8k 需要 --flickr_captions_file")
            cap_map = read_flickr8k_captions(args.flickr_captions_file)
            # 摊平 + 选择
            all_caps = []
            rng = np.random.default_rng()
            for _, caps in cap_map.items():
                if not caps:
                    continue
                all_caps.append(caps[0] if args.flickr_caption_pick == "first"
                                else caps[int(rng.integers(0, len(caps)))])
            if not all_caps:
                raise RuntimeError("Flickr8k captions 为空")
            texts = [all_caps[i % len(all_caps)] for i in range(args.num_samples)]

        embs = enc.encode_texts(texts, batch_size=args.batch_size)

    else:
        # 多模态：文本+图片（pairlist）
        if args.data_source == "random":
            images, texts = make_random_images_and_texts(args.num_samples)
        else:
            if not args.flickr_images_dir or not args.flickr_captions_file:
                raise ValueError("--data_source flickr8k 需要 --flickr_images_dir 与 --flickr_captions_file")
            images, texts = load_flickr8k_pairs(
                args.flickr_images_dir, args.flickr_captions_file,
                total=args.num_samples, pick_caption=args.flickr_caption_pick
            )

        embs = enc.encode_v1_multimodal_pairlist(
            texts=texts,
            images=images,
            batch_size=args.batch_size,
            image_transport=args.image_transport,
        )

    elapsed = time.time() - t0
    print("\n==== Summary ====")
    print(f"mode={args.mode}, data_source={args.data_source}, num_samples={args.num_samples}, batch_size={args.batch_size}")
    print(f"shape={tuple(embs.shape)}  (dim={embs.shape[1] if embs.numel() else 0})")
    print(f"time(s)={elapsed:.4f}, throughput(samples/s)={args.num_samples / elapsed:.2f}")


if __name__ == "__main__":
    main()
