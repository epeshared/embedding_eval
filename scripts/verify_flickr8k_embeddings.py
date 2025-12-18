#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from typing import Tuple

import torch


def _load_pt(path: str) -> Tuple[list, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or "embeddings" not in obj:
        raise ValueError(f"Unexpected format in {path}: expected dict with key 'embeddings'")
    embs = obj["embeddings"]
    if not isinstance(embs, torch.Tensor):
        raise ValueError(f"Unexpected embeddings type in {path}: {type(embs)}")

    # optional metadata
    items = obj.get("images") or obj.get("texts") or []
    if not isinstance(items, list):
        items = []
    return items, embs


def _recall_at_k(sim: torch.Tensor, gt: torch.Tensor, k: int) -> float:
    # sim: [N, M], gt: [N] integer indices in [0, M)
    topk = sim.topk(k=min(k, sim.size(1)), dim=1).indices
    hit = (topk == gt.view(-1, 1)).any(dim=1)
    return hit.float().mean().item()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Verify Flickr8k text/image embeddings: sanity checks + retrieval Recall@K.\n"
            "Input files should be produced by main.py via --dump-img-emb and --dump-txt-emb."
        )
    )
    ap.add_argument("--img-pt", required=True, help="Path to torch.save output with image embeddings")
    ap.add_argument("--txt-pt", required=True, help="Path to torch.save output with text embeddings")
    ap.add_argument(
        "--captions-per-image",
        type=int,
        default=1,
        help="Captions per image used when generating embeddings (default: 1)",
    )
    ap.add_argument("--k", type=str, default="1,5,10", help="Comma-separated K list for Recall@K")
    ap.add_argument("--max-n", type=int, default=-1, help="Optional: only evaluate on first N images")
    args = ap.parse_args()

    _, img_emb = _load_pt(args.img_pt)
    _, txt_emb = _load_pt(args.txt_pt)

    cap_per_img = max(1, int(args.captions_per_image))

    if img_emb.ndim != 2 or txt_emb.ndim != 2:
        raise ValueError(f"Expected 2D tensors, got img={tuple(img_emb.shape)} txt={tuple(txt_emb.shape)}")

    if args.max_n and args.max_n > 0:
        n = min(int(args.max_n), img_emb.size(0))
        img_emb = img_emb[:n]
        txt_emb = txt_emb[: n * cap_per_img]

    n_img, dim_i = img_emb.shape
    n_txt, dim_t = txt_emb.shape

    print(f"[Info] img_emb: shape={tuple(img_emb.shape)}")
    print(f"[Info] txt_emb: shape={tuple(txt_emb.shape)}")

    if dim_i != dim_t:
        raise ValueError(f"Embedding dims differ: image={dim_i} text={dim_t}")

    expected_txt = n_img * cap_per_img
    if n_txt != expected_txt:
        raise ValueError(
            f"Text count mismatch: n_txt={n_txt}, expected n_img*captions_per_image={expected_txt}. "
            "Pass the same --captions-per-image used during embedding generation."
        )

    def _stats(name: str, x: torch.Tensor) -> None:
        nan = torch.isnan(x).any().item()
        inf = torch.isinf(x).any().item()
        norms = torch.linalg.norm(x.float(), dim=1)
        print(
            f"[Sanity] {name}: nan={nan} inf={inf} "
            f"norm(mean/std/min/max)={norms.mean().item():.4f}/{norms.std().item():.4f}/"
            f"{norms.min().item():.4f}/{norms.max().item():.4f}"
        )

    _stats("image", img_emb)
    _stats("text", txt_emb)

    # Normalize just in case (some backends already do)
    img_n = torch.nn.functional.normalize(img_emb.float(), p=2, dim=1)
    txt_n = torch.nn.functional.normalize(txt_emb.float(), p=2, dim=1)

    # text -> image: each text corresponds to one image index (t // cap_per_img)
    sim_t2i = txt_n @ img_n.t()  # [n_txt, n_img]
    gt_t2i = torch.arange(n_txt, dtype=torch.long) // cap_per_img

    # image -> text: define the "correct" caption as the best among its captions
    # We compute ranks by using max similarity over captions for each image.
    sim_i2t = img_n @ txt_n.t()  # [n_img, n_txt]

    # Diagonal-ish sanity: paired similarities should exceed random on average
    paired = sim_t2i[torch.arange(n_txt), gt_t2i]
    rand = sim_t2i.flatten()
    print(f"[Sanity] paired_sim mean/std={paired.mean().item():.4f}/{paired.std().item():.4f}")
    print(f"[Sanity] all_sim    mean/std={rand.mean().item():.4f}/{rand.std().item():.4f}")

    ks = [int(x) for x in args.k.split(",") if x.strip()]

    print("\n[Retrieval] Text -> Image")
    for k in ks:
        r = _recall_at_k(sim_t2i, gt_t2i, k)
        print(f"  R@{k}: {r:.4f}")

    print("\n[Retrieval] Image -> Text")
    # For each image, the correct set is texts in [i*cap_per_img, (i+1)*cap_per_img)
    # We compute Recall@K by checking if ANY correct caption is in topK.
    topk = sim_i2t.topk(k=min(max(ks), sim_i2t.size(1)), dim=1).indices  # [n_img, Kmax]
    for k in ks:
        tk = topk[:, : min(k, topk.size(1))]
        # build per-image correct ranges
        base = (torch.arange(n_img, dtype=torch.long) * cap_per_img).view(-1, 1)
        corr = base + torch.arange(cap_per_img, dtype=torch.long).view(1, -1)  # [n_img, cap_per_img]
        hit = (tk.unsqueeze(-1) == corr.unsqueeze(1)).any(dim=(1, 2))
        print(f"  R@{k}: {hit.float().mean().item():.4f}")


if __name__ == "__main__":
    main()
