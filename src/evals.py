import time, os
from typing import Dict, List
import torch
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from .utils import batched_cosine_similarity, l2_normalize

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
