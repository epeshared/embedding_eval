#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Any, List, Optional

import torch


def _as_tensor(x: Any) -> Optional[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return x
    return None


def _load_any(path: Path) -> Any:
    return torch.load(path, map_location="cpu")


def _summarize_embeddings(x: torch.Tensor, name: str = "embeddings") -> None:
    x = x.detach().cpu()
    print(f"[{name}] dtype={x.dtype} shape={tuple(x.shape)}")
    if x.ndim != 2:
        print(f"[{name}] non-2D tensor; skip vector stats")
        return

    xf = x.float()
    any_nan = torch.isnan(xf).any().item()
    any_inf = torch.isinf(xf).any().item()
    norms = torch.linalg.norm(xf, dim=1)
    per_dim_std = xf.std(dim=0)

    print(f"[{name}] nan={any_nan} inf={any_inf}")
    print(
        f"[{name}] L2 norm mean/std/min/max="
        f"{norms.mean().item():.6f}/{norms.std().item():.6f}/{norms.min().item():.6f}/{norms.max().item():.6f}"
    )
    print(
        f"[{name}] per-dim std mean/min/max="
        f"{per_dim_std.mean().item():.6f}/{per_dim_std.min().item():.6f}/{per_dim_std.max().item():.6f}"
    )

    # Approx uniqueness via random projections (fast)
    g = torch.Generator().manual_seed(0)
    proj = torch.randn((xf.size(1), 8), generator=g)
    sig = (torch.nn.functional.normalize(xf, p=2, dim=1) @ proj).numpy().round(6)
    import numpy as np

    uniq = np.unique(sig, axis=0).shape[0]
    print(f"[{name}] approx-unique-signatures={uniq}/{xf.size(0)}")


def _print_list_head(items: List[Any], label: str, n: int) -> None:
    if not items:
        print(f"[{label}] empty")
        return
    print(f"[{label}] count={len(items)}")
    for i, v in enumerate(items[:n]):
        s = str(v)
        if len(s) > 200:
            s = s[:200] + "..."
        print(f"  {i}: {s}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect a torch .pt file (torch.save output) used in this repo.")
    ap.add_argument("path", help="Path to .pt file")
    ap.add_argument("--head", type=int, default=3, help="How many metadata items to print")
    ap.add_argument("--rows", type=int, default=2, help="How many embedding rows to print (first rows)")
    ap.add_argument("--cols", type=int, default=8, help="How many embedding dims to print per row")
    ap.add_argument(
        "--idx",
        type=str,
        default="",
        help="Comma-separated row indices to print (overrides --rows). Example: --idx 0,1,42",
    )
    ap.add_argument(
        "--all-cols",
        action="store_true",
        help="Print the full embedding vector for selected rows (can be very long).",
    )
    ap.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Decimal places for printing embedding values (default: 6)",
    )
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    obj = _load_any(path)
    print(f"[File] {path}")
    print(f"[Top] type={type(obj)}")

    if isinstance(obj, dict):
        print(f"[Dict] keys={list(obj.keys())}")
        # common schema in this repo: {images/texts: [...], embeddings: Tensor}
        if "images" in obj and isinstance(obj["images"], list):
            _print_list_head(obj["images"], "images", args.head)
        if "texts" in obj and isinstance(obj["texts"], list):
            _print_list_head(obj["texts"], "texts", args.head)

        emb = _as_tensor(obj.get("embeddings"))
        if emb is not None:
            _summarize_embeddings(emb, "embeddings")
            if emb.ndim == 2:
                if args.idx.strip():
                    try:
                        sel = [int(x.strip()) for x in args.idx.split(",") if x.strip()]
                    except Exception:
                        raise SystemExit("--idx must be a comma-separated list of integers, e.g. --idx 0,1,42")
                else:
                    sel = list(range(min(max(0, int(args.rows)), emb.size(0))))

                if not sel:
                    return

                fmt = "{:." + str(int(args.precision)) + "f}"
                cols = emb.size(1) if args.all_cols else min(max(1, int(args.cols)), emb.size(1))

                header = "selected" if args.idx.strip() else f"first {len(sel)}"
                print(f"[embeddings] {header} rows, {'all' if args.all_cols else cols} dims:")
                for i in sel:
                    if i < 0 or i >= emb.size(0):
                        print(f"  {i}: <out of range>")
                        continue
                    v = emb[i, :cols].tolist()
                    print(f"  {i}: {[fmt.format(float(x)) for x in v]}")
        else:
            # fall back: show any tensor-like fields
            tensor_keys = [k for k, v in obj.items() if isinstance(v, torch.Tensor)]
            if tensor_keys:
                print(f"[Dict] tensor keys={tensor_keys}")
                for k in tensor_keys:
                    t = obj[k]
                    print(f"  - {k}: dtype={t.dtype} shape={tuple(t.shape)}")

    elif isinstance(obj, torch.Tensor):
        _summarize_embeddings(obj, "tensor")

    else:
        # unknown structure
        s = str(obj)
        if len(s) > 4000:
            s = s[:4000] + "..."
        print("[Preview]", s)


if __name__ == "__main__":
    main()
