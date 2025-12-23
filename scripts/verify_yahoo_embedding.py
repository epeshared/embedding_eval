#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any


@dataclass
class Rec:
    key: str
    idx: Optional[int]
    text: Optional[str]
    emb: List[float]


def iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON decode error in {path}:{ln}: {e}\nLine={line[:200]}...")


def as_float_list(x: Any) -> List[float]:
    if not isinstance(x, list):
        raise TypeError(f"embedding is not a list, got {type(x)}")
    # ensure float
    out = []
    for v in x:
        try:
            out.append(float(v))
        except Exception as e:
            raise TypeError(f"embedding element not float-convertible: {v} ({type(v)}): {e}")
    return out


def load_map(path: str | Path, key_field: str) -> Dict[str, Rec]:
    m: Dict[str, Rec] = {}
    for obj in iter_jsonl(path):
        idx = obj.get("idx", None)
        text = obj.get("text", None)

        if key_field == "text":
            if text is None:
                continue
            key = str(text)
        elif key_field == "idx":
            if idx is None:
                continue
            key = str(idx)
        else:
            raise ValueError("key_field must be 'text' or 'idx'")

        emb = as_float_list(obj.get("embedding"))
        m[key] = Rec(key=key, idx=idx if isinstance(idx, int) else None, text=text, emb=emb)
    return m


def dot(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dim mismatch: {len(a)} vs {len(b)}")
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


def l2(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dim mismatch: {len(a)} vs {len(b)}")
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return math.sqrt(s)


def norm(a: List[float]) -> float:
    return math.sqrt(dot(a, a))


def cosine(a: List[float], b: List[float]) -> float:
    na = norm(a)
    nb = norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return dot(a, b) / (na * nb)


def summarize(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"count": 0, "min": float("nan"), "max": float("nan"), "avg": float("nan")}
    return {
        "count": float(len(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "avg": float(sum(vals) / len(vals)),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Compare embeddings between two jsonl files by text/idx key and compute similarity."
    )
    ap.add_argument("--a", required=True, help="jsonl file A (baseline)")
    ap.add_argument("--b", required=True, help="jsonl file B (to compare)")
    ap.add_argument("--key", choices=["text", "idx"], default="text", help="match key field")
    ap.add_argument("--metric", choices=["cosine", "dot", "l2"], default="cosine", help="similarity/distance metric")
    ap.add_argument("--threshold", type=float, default=0.999, help="flag items with cosine/dot < threshold (or l2 > threshold)")
    ap.add_argument("--topk", type=int, default=20, help="show top-k worst pairs")
    ap.add_argument("--out", default="", help="optional output csv path")
    args = ap.parse_args()

    A = load_map(args.a, args.key)
    B = load_map(args.b, args.key)

    keysA = set(A.keys())
    keysB = set(B.keys())
    common = sorted(keysA & keysB)
    onlyA = sorted(keysA - keysB)
    onlyB = sorted(keysB - keysA)

    print(f"[Load] A={args.a} items={len(A)}")
    print(f"[Load] B={args.b} items={len(B)}")
    print(f"[Match] common={len(common)} onlyA={len(onlyA)} onlyB={len(onlyB)}")

    if len(common) == 0:
        raise SystemExit("No common keys found. Try --key idx or check your files.")

    # metric function
    if args.metric == "cosine":
        f = cosine
        worse_sort = lambda x: x[0]  # smaller is worse
        is_bad = lambda v: (not math.isnan(v)) and (v < args.threshold)
    elif args.metric == "dot":
        f = dot
        worse_sort = lambda x: x[0]  # smaller is worse
        is_bad = lambda v: v < args.threshold
    else:  # l2
        f = l2
        worse_sort = lambda x: -x[0]  # larger is worse (sort descending)
        is_bad = lambda v: v > args.threshold

    results: List[Tuple[float, str]] = []
    bad: List[Tuple[float, str]] = []

    rows_for_csv: List[Tuple[str, Optional[int], str, int, float]] = []
    # key, idx, text_snip, dim, score

    for k in common:
        ra = A[k]
        rb = B[k]
        score = f(ra.emb, rb.emb)
        results.append((score, k))
        if is_bad(score):
            bad.append((score, k))

        text = ra.text if ra.text is not None else ""
        text_snip = text.replace("\n", " ")[:120]
        rows_for_csv.append((k, ra.idx, text_snip, len(ra.emb), float(score)))

    scores = [s for s, _ in results if not math.isnan(s)]
    stats = summarize(scores)
    print(f"[Stats:{args.metric}] count={int(stats['count'])} min={stats['min']:.6f} max={stats['max']:.6f} avg={stats['avg']:.6f}")
    print(f"[Flag] threshold={args.threshold} flagged={len(bad)}")

    # show worst topk
    if args.metric in ("cosine", "dot"):
        worst = sorted(results, key=lambda x: x[0])[: args.topk]
        print(f"\n[Worst {args.topk}] (smaller={args.metric} worse)")
    else:
        worst = sorted(results, key=lambda x: x[0], reverse=True)[: args.topk]
        print(f"\n[Worst {args.topk}] (larger=l2 worse)")

    for score, k in worst:
        text = A[k].text or ""
        print(f"{args.metric}={score:.6f} key={k} dim={len(A[k].emb)} text={text[:80]!r}")

    # optionally write csv
    if args.out:
        import csv
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", newline="", encoding="utf-8") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["key", "idx", "text_snip", "dim", args.metric])
            for r in rows_for_csv:
                w.writerow(r)
        print(f"\n[CSV] wrote {len(rows_for_csv)} rows -> {args.out}")

    # optionally show missing samples
    if onlyA:
        print(f"\n[Only in A] showing up to 10: {onlyA[:10]}")
    if onlyB:
        print(f"\n[Only in B] showing up to 10: {onlyB[:10]}")


if __name__ == "__main__":
    main()
