#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import mimetypes
import os
from typing import Any, Dict, List

import torch

try:
    import requests  # type: ignore
except Exception as e:
    raise RuntimeError(f"requests is required: {e}")


def _image_to_repr(img: str, transport: str) -> str:
    transport = (transport or "data-url").lower()

    if transport == "path/url":
        return img

    s = img.strip()

    if transport == "base64":
        if s.startswith("data:"):
            comma = s.find(",")
            return s[comma + 1 :] if comma >= 0 else s
        if s.startswith("http://") or s.startswith("https://"):
            return s
        if os.path.exists(s) and os.path.isfile(s):
            with open(s, "rb") as f:
                return base64.b64encode(f.read()).decode("ascii")
        return s

    # data-url
    if s.startswith("data:") or s.startswith("http://") or s.startswith("https://"):
        return s
    if os.path.exists(s) and os.path.isfile(s):
        mime, _ = mimetypes.guess_type(s)
        mime = mime or "application/octet-stream"
        with open(s, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    return s


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    a = a / (a.norm(p=2) + 1e-12)
    b = b / (b.norm(p=2) + 1e-12)
    return float((a * b).sum().item())


def main():
    ap = argparse.ArgumentParser(description="Quick sanity test for SGLang /v1/embeddings multimodal image embeddings")
    ap.add_argument("--url", type=str, default="http://127.0.0.1:30000", help="SGLang server base URL")
    ap.add_argument("--model", type=str, default="default", help="Model name/path (sent as OpenAI 'model')")
    ap.add_argument("--image1", type=str, required=True)
    ap.add_argument("--image2", type=str, required=True)
    ap.add_argument("--transport", type=str, default="data-url", choices=["data-url", "base64", "path/url"])
    ap.add_argument("--text", type=str, default="padding")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--dims", type=int, default=10)
    args = ap.parse_args()

    img1 = _image_to_repr(args.image1, args.transport)
    img2 = _image_to_repr(args.image2, args.transport)

    payload: Dict[str, Any] = {
        "model": args.model,
        "input": [
            {"text": args.text, "image": img1},
            {"text": args.text, "image": img2},
        ],
    }

    url = args.url.rstrip("/") + "/v1/embeddings"
    resp = requests.post(url, json=payload, timeout=args.timeout)
    resp.raise_for_status()
    data = resp.json()

    rows: List[Dict[str, Any]] = data.get("data", [])
    rows = sorted(rows, key=lambda r: r.get("index", 0))
    if len(rows) < 2:
        raise RuntimeError(f"Unexpected response: {data}")

    e1 = torch.tensor(rows[0]["embedding"], dtype=torch.float32)
    e2 = torch.tensor(rows[1]["embedding"], dtype=torch.float32)

    print(f"[OK] transport={args.transport} url={url}")
    print(f"cos(e1,e2)={_cos(e1,e2):.6f}")
    print(f"first {args.dims} dims e1={e1[:args.dims].tolist()}")
    print(f"first {args.dims} dims e2={e2[:args.dims].tolist()}")


if __name__ == "__main__":
    main()
