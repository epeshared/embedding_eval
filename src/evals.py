import time, os
from typing import Dict, List
import torch
from sklearn.metrics import accuracy_score, f1_score
from .utils import batched_cosine_similarity, l2_normalize


def _read_flickr8k_captions(token_txt: str) -> Dict[str, List[str]]:
    """Flickr8k.token.txt -> {filename: [captions...]}"""
    mp: Dict[str, List[str]] = {}
    with open(token_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            left, cap = line.split("\t", 1)
            img, _ = left.split("#", 1)
            mp.setdefault(img, []).append(cap)
    return mp


@torch.inference_mode()
def eval_flickr8k_perf(
    encoder,
    images_dir: str,
    captions_file: str,
    batch_size: int = 128,
    max_images: int = -1,
    captions_per_image: int = 1,
    modality: str = "both",
    dump_img_emb: str = "",
    dump_txt_emb: str = "",
    output_csv: str = "",
    output_jsonl: str = "",
) -> Dict:
    """Throughput benchmark on Flickr8k.

    modality:
      - "both": run text embedding and image embedding (default; backwards compatible)
      - "text": run caption text embedding only (does NOT require images to exist)
      - "image": run image embedding only

    Requires encoder to provide:
      - for text: encode(texts: List[str], batch_size=...)
      - for image: encode_images(images: List[...], batch_size=...)
    """

    modality = (modality or "both").lower().strip()
    if modality not in ("both", "text", "image"):
        raise ValueError("modality must be one of: both|text|image")

    if modality in ("both", "text") and not hasattr(encoder, "encode"):
        raise RuntimeError("Encoder does not support text embedding (missing encode).")
    if modality in ("both", "image") and not hasattr(encoder, "encode_images"):
        raise RuntimeError("Encoder does not support image embedding (missing encode_images).")

    cap_map = _read_flickr8k_captions(captions_file)
    if not cap_map:
        raise RuntimeError("Flickr8k captions are empty.")

    image_paths: List[str] = []
    texts: List[str] = []
    per_img = max(1, int(captions_per_image))

    for fn, caps in sorted(cap_map.items()):
        if modality in ("both", "image"):
            p = os.path.join(images_dir, fn)
            if not os.path.exists(p):
                continue
            image_paths.append(p)

        if modality in ("both", "text"):
            if not caps:
                take = [""] * per_img
            else:
                take = caps[:per_img] if len(caps) >= per_img else (caps + caps[: (per_img - len(caps))])
            texts.extend(take)

        if max_images > 0:
            if modality in ("both", "image"):
                if len(image_paths) >= max_images:
                    break
            else:
                if (len(texts) // per_img) >= max_images:
                    break

    if modality in ("both", "image") and not image_paths:
        raise RuntimeError("No Flickr8k images found matching captions file.")
    if modality in ("both", "text") and not texts:
        raise RuntimeError("No Flickr8k captions found.")

    n_images = len(image_paths)
    n_texts = len(texts)
    print(
        f"[Eval] Flickr8k perf: modality={modality} images={n_images}, texts={n_texts} "
        f"(captions_per_image={per_img})"
    )

    txt_emb = None
    img_emb = None

    if modality in ("both", "text"):
        t0 = time.time()
        txt_emb = encoder.encode(texts, batch_size=batch_size)
        t1 = time.time()
        txt_time = t1 - t0
        txt_qps = n_texts / txt_time if txt_time > 0 else 0.0
        txt_batches = (n_texts + batch_size - 1) // batch_size
        txt_avg_batch_sec = (txt_time / txt_batches) if txt_batches > 0 else 0.0
        txt_shape = tuple(txt_emb.shape)
    else:
        txt_time = 0.0
        txt_qps = 0.0
        txt_batches = 0
        txt_avg_batch_sec = 0.0
        txt_shape = None

    if modality in ("both", "image"):
        v0 = time.time()
        img_emb = encoder.encode_images(image_paths, batch_size=batch_size)
        v1 = time.time()
        img_time = v1 - v0
        img_qps = n_images / img_time if img_time > 0 else 0.0
        img_batches = (n_images + batch_size - 1) // batch_size
        img_avg_batch_sec = (img_time / img_batches) if img_batches > 0 else 0.0
        img_shape = tuple(img_emb.shape)
    else:
        img_time = 0.0
        img_qps = 0.0
        img_batches = 0
        img_avg_batch_sec = 0.0
        img_shape = None

    print(
        f"[flickr8k] text: n={n_texts} time={txt_time:.3f}s QPS={txt_qps:.2f} "
        f"avg_batch={txt_avg_batch_sec:.4f}s ({txt_avg_batch_sec*1000:.2f}ms) shape={txt_shape} | "
        f"image: n={n_images} time={img_time:.3f}s QPS={img_qps:.2f} "
        f"avg_batch={img_avg_batch_sec:.4f}s ({img_avg_batch_sec*1000:.2f}ms) shape={img_shape}"
    )

    if dump_txt_emb and txt_emb is not None:
        os.makedirs(os.path.dirname(dump_txt_emb) or ".", exist_ok=True)
        torch.save({"texts": texts, "embeddings": txt_emb}, dump_txt_emb)
        print(f"[Dump] text embeddings -> {dump_txt_emb}")
    if dump_img_emb and img_emb is not None:
        os.makedirs(os.path.dirname(dump_img_emb) or ".", exist_ok=True)
        torch.save({"images": image_paths, "embeddings": img_emb}, dump_img_emb)
        print(f"[Dump] image embeddings -> {dump_img_emb}")

    rec = {
        "dataset": "flickr8k",
        "modality": modality,
        "n_images": n_images,
        "n_texts": n_texts,
        "captions_per_image": per_img,
        "batch_size": batch_size,
        "text_time_sec": round(txt_time, 6),
        "text_qps": txt_qps,
        "text_batches": int(txt_batches),
        "text_avg_batch_sec": round(txt_avg_batch_sec, 6),
        "image_time_sec": round(img_time, 6),
        "image_qps": img_qps,
        "image_batches": int(img_batches),
        "image_avg_batch_sec": round(img_avg_batch_sec, 6),
    }

    if output_jsonl:
        try:
            from . import utils as _utils

            _utils.append_jsonl(output_jsonl, rec, {})
        except Exception:
            import json

            os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
            with open(output_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[Save] jsonl -> {output_jsonl}")

    if output_csv:
        try:
            from . import utils as _utils

            _utils.append_csv(output_csv, rec, {})
        except Exception:
            import csv

            os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
            new = (not os.path.exists(output_csv)) or os.path.getsize(output_csv) == 0
            with open(output_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rec.keys()))
                if new:
                    w.writeheader()
                w.writerow(rec)
        print(f"[Save] csv -> {output_csv}")

    return rec

@torch.inference_mode()
def eval_dataset_text_pairs(encoder, dataset_key: str, split: str="validation",
                            batch_size: int=128, max_samples: int=-1, threshold: float=0.5) -> Dict:
    from .datasets import load_pairs
    s1, s2, labels = load_pairs(dataset_key, split, max_samples)
    n = len(labels); print(f"[Eval] {dataset_key}: {n} samples")
    t0 = time.time(); e1 = encoder.encode(s1, batch_size=batch_size); e2 = encoder.encode(s2, batch_size=batch_size); t1 = time.time()
    sims = batched_cosine_similarity(e1, e2); preds = (sims > threshold).to(torch.int64)
    acc = accuracy_score(labels, preds.numpy().tolist()); f1 = f1_score(labels, preds.numpy().tolist())
    t2 = time.time(); encode_time = t1-t0; score_time = t2-t1; total_time = t2-t0; qps = n/total_time if total_time>0 else 0.0
    print(f"[{dataset_key}] acc={acc:.4f} f1={f1:.4f} | encode={encode_time:.3f}s score={score_time:.3f}s total={total_time:.3f}s | QPS={qps:.2f}")
    return {"dataset":dataset_key,"split":split,"n_samples":n,"acc":acc,"f1":f1,
            "encode_time":round(encode_time,6),"score_time":round(score_time,6),
            "total_time":round(total_time,6),"qps":qps}

@torch.inference_mode()
def eval_food101_zeroshot(clip, split: str="validation",
                          max_samples: int=-1, batch_size_img: int=128,
                          prompt_template: str = "a photo of a {}",
                          dump_img_emb: str = "", dump_txt_emb: str = "") -> Dict:
    print(f"[Eval] Food-101 ({split})")
    # Lazy import: HF datasets pulls in `multiprocess`, which can emit noisy shutdown
    # warnings on some Python versions/environments. Only import when needed.
    from datasets import load_dataset
    ds = load_dataset("food101", split=split)
    label_names = ds.features["label"].names
    prompts = [prompt_template.format(name.replace("_", " ")) for name in label_names]

    t0 = time.time()
    txt_feats = clip.encode_text(prompts, batch_size=256)  # [C, D]
    t1 = time.time()

    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))
    images = [r["image"] for r in ds]
    labels = torch.tensor([int(r["label"]) for r in ds], dtype=torch.long)
    v0 = time.time()
    img_feats = clip.encode_images(images, batch_size=batch_size_img)  # [N, D]
    v1 = time.time()

    logits = img_feats @ txt_feats.T  # [N, C]
    top1 = logits.argmax(dim=1)
    top5 = torch.topk(logits, k=5, dim=1).indices

    top1_acc = (top1 == labels).float().mean().item()
    top5_acc = (top5 == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    total = len(labels)
    encode_time_text = t1 - t0
    encode_time_img  = v1 - v0
    total_time = (t1 - t0) + (v1 - v0)
    qps = total / total_time if total_time > 0 else 0.0

    print(f"[food101] top1={top1_acc:.4f} top5={top5_acc:.4f} | "
          f"text_enc={encode_time_text:.3f}s img_enc={encode_time_img:.3f}s total={total_time:.3f}s | QPS={qps:.2f}")

    if dump_txt_emb:
        os.makedirs(os.path.dirname(dump_txt_emb) or ".", exist_ok=True)
        torch.save({"labels": label_names, "prompts": prompts, "embeddings": txt_feats}, dump_txt_emb)
        print(f"[Dump] text embeddings -> {dump_txt_emb}")
    if dump_img_emb:
        os.makedirs(os.path.dirname(dump_img_emb) or ".", exist_ok=True)
        torch.save({"labels": labels, "embeddings": img_feats}, dump_img_emb)
        print(f"[Dump] image embeddings -> {dump_img_emb}")

    return {
        "dataset": "food101",
        "split": split,
        "n_samples": total,
        "top1": top1_acc,
        "top5": top5_acc,
        "encode_time_text": round(encode_time_text, 6),
        "encode_time_img": round(encode_time_img, 6),
        "total_time": round(total_time, 6),
        "qps": qps
    }

@torch.inference_mode()
def eval_unlabeled_texts(encoder, texts, batch_size: int=128, dump_emb_path: str=""):
    """对无标签文本列表进行编码，打印吞吐，支持将向量保存到文件。"""
    import os, time, torch
    if not texts:
        print("[Eval] No texts to encode.")
        return {"n_samples": 0, "encode_time": 0.0, "qps": 0.0}
    n = len(texts)
    t0 = time.time()
    embs = encoder.encode(texts, batch_size=batch_size)
    t1 = time.time()
    encode_time = t1 - t0
    qps = n / encode_time if encode_time > 0 else 0.0
    print(f"[unlabeled] n={n} encode={encode_time:.3f}s | QPS={qps:.2f}")
    if dump_emb_path:
        os.makedirs(os.path.dirname(dump_emb_path) or ".", exist_ok=True)
        torch.save({ "texts": texts, "embeddings": embs }, dump_emb_path)
        print(f"[Dump] embeddings -> {dump_emb_path}")
    return {"n_samples": n, "encode_time": round(encode_time, 6), "qps": qps}
