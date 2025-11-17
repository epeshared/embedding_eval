#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified embedding evaluation entry.
Backends: transformers, sglang, clip, vllm_openai (vLLM OpenAI-compatible), vllm (native LLM.embed), llamacpp(optional placeholder)
Tasks: pairwise text classification (LCQMC/AFQMC/PAWS-X zh etc.), Food-101 zero-shot, unlabeled Yahoo JSONL encode, MTEB bridge.
This file tries to import helper modules from either `src.*` or local files for robustness.
"""

import os
import sys
import time
import json
import argparse
from typing import List, Optional, Dict, Any
import torch

# ---- Robust imports: prefer src.*, fall back to local modules ----
def _try_import():
    mods = {}
    # utils
    try:
        from src import utils as _utils
        mods['utils'] = _utils
    except Exception:
        import utils as _utils
        mods['utils'] = _utils

    # evals
    try:
        from src import evals as _evals
        mods['evals'] = _evals
    except Exception:
        import evals as _evals
        mods['evals'] = _evals

    # datasets (NEW)
    try:
        from src import datasets as _datasets
        mods['datasets'] = _datasets
    except Exception:
        try:
            import datasets as _datasets
            mods['datasets'] = _datasets
        except Exception:
            mods['datasets'] = None

    # mteb bridge (optional)
    try:
        from src import mteb_bridge as _mteb
        mods['mteb'] = _mteb
    except Exception:
        try:
            import mteb_bridge as _mteb
            mods['mteb'] = _mteb
        except Exception:
            mods['mteb'] = None

    # sglang backend (optional)
    try:
        from src.backends.sglang_backend import SGLangEncoder as _SGL
        mods['sgl_cls'] = _SGL
    except Exception:
        try:
            from sglang_backend import SGLangEncoder as _SGL
            mods['sgl_cls'] = _SGL
        except Exception:
            mods['sgl_cls'] = None

    # vLLM OpenAI backend (optional)
    try:
        from src.backends.vllm_openai_backend import VllmOpenAIEncoder as _VLLM_OAI
        mods['vllm_oai_cls'] = _VLLM_OAI
    except Exception:
        try:
            from vllm_openai_backend import VllmOpenAIEncoder as _VLLM_OAI
            mods['vllm_oai_cls'] = _VLLM_OAI
        except Exception:
            mods['vllm_oai_cls'] = None

    # vLLM native backend (NEW)
    try:
        from src.backends.vllm_backend import VLLMEncoder as _VLLM
        mods['vllm_cls'] = _VLLM
    except Exception:
        try:
            from vllm_backend import VLLMEncoder as _VLLM
            mods['vllm_cls'] = _VLLM
        except Exception:
            mods['vllm_cls'] = None

    return mods

mods = _try_import()
utils = mods['utils']
evals = mods['evals']
datasets_mod = mods['datasets']   # <- datasets 模块（包含 load_yahoo_answers_jsonl）
mteb = mods['mteb']
SGLangEncoder = mods['sgl_cls']
VllmOpenAIEncoder = mods['vllm_oai_cls']
VLLMEncoder = mods['vllm_cls']

# ---- Minimal Transformers encoder (mean pooling) ----
class HFEncoder:
    def __init__(self, model: str, device: str = "cpu",
                 use_ipex: bool = False, amp: str = "auto",
                 trust_remote_code: bool = True, max_length: int = 512,
                 offline: bool = False, hf_endpoint: str = ""):
        from transformers import AutoModel, AutoTokenizer
        if offline:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint
            os.environ["HF_HUB_BASE_URL"] = hf_endpoint

        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(model, trust_remote_code=trust_remote_code)
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.max_length = max_length
        self.amp = amp.lower()
        self.use_ipex = bool(use_ipex)

        # optional IPEX
        if self.use_ipex and device == "cpu":
            try:
                import intel_extension_for_pytorch as ipex  # noqa: F401
                print("[Init] IPEX enabled")
            except Exception as e:
                print(f"[Warn] IPEX import failed: {e}")

    @torch.inference_mode()
    def encode(self, texts: List[str], batch_size: int = 128):
        import torch
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            toks = self.tok(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            toks = {k: v.to(self.device) for k, v in toks.items()}
            with torch.autocast(device_type=("cpu" if self.device=="cpu" else "cuda"),
                                dtype=torch.bfloat16 if self.amp in ("bf16","auto") else torch.float16,
                                enabled=(self.amp!="off")):
                out_hidden = self.model(**toks).last_hidden_state  # [B, T, H]
                attn = toks["attention_mask"].unsqueeze(-1).float()
                summed = (out_hidden * attn).sum(dim=1)
                denom = attn.sum(dim=1).clamp(min=1e-6)
                emb = summed / denom                     # mean pooling
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)  # L2
                out.append(emb.to("cpu"))
        return torch.cat(out, dim=0)

# ---- CLIP encoder (text/image) for Food-101 ----
class CLIPEncoder:
    def __init__(self, model: str, device: str="cpu", use_ipex: bool=False, amp: str="auto", trust_remote_code: bool=True):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(model, trust_remote_code=trust_remote_code).to(device)
        self.processor = CLIPProcessor.from_pretrained(model, trust_remote_code=trust_remote_code)
        self.device = device
        self.amp = amp

    @torch.inference_mode()
    def encode_text(self, texts: List[str], batch_size: int=64):
        import torch
        outs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.autocast(device_type=("cpu" if self.device=="cpu" else "cuda"),
                                dtype=torch.bfloat16, enabled=(self.amp!="off")):
                embs = self.model.get_text_features(**inputs)
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            outs.append(embs.to("cpu"))
        return torch.cat(outs, dim=0)

    @torch.inference_mode()
    def encode_images(self, images, batch_size: int=32):
        import torch
        outs = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with torch.autocast(device_type=("cpu" if self.device=="cpu" else "cuda"),
                                dtype=torch.bfloat16, enabled=(self.amp!="off")):
                embs = self.model.get_image_features(**inputs)
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            outs.append(embs.to("cpu"))
        return torch.cat(outs, dim=0)

# ---------------- Yahoo (JSONL) via datasets.load_yahoo_answers_jsonl ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Embedding evaluation launcher (transformers / SGLang / CLIP / vllm_openai(vLLM) / vllm(native))")
    p.add_argument("--backend", type=str, default="transformers",
                   choices=["transformers","sglang","clip","vllm_openai","vllm","llamacpp"])
    p.add_argument("--model", type=str, default="BAAI/bge-large-zh-v1.5")

    # datasets / tasks
    p.add_argument("--datasets", type=str, default="", help="Comma list: LCQMC,AFQMC,PAWSX-zh,FOOD101,MTEB,...")
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--output-csv", type=str, default="")
    p.add_argument("--output-jsonl", type=str, default="")

    # unlabeled (Yahoo JSONL)
    p.add_argument("--yahoo-jsonl", type=str, default="")
    p.add_argument("--yahoo-mode", type=str, default="q", choices=["q","q+a"])
    p.add_argument("--yahoo-max", type=int, default=10000)
    p.add_argument("--dump-emb", type=str, default="")
    p.add_argument("--dump-img-emb", type=str, default="")
    p.add_argument("--dump-txt-emb", type=str, default="")

    # device / perf
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--use-ipex", type=str, default="False")
    p.add_argument("--amp", type=str, default="auto", choices=["off","auto","fp16","bf16"])
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--offline", type=str, default="False")
    p.add_argument("--hf-endpoint", type=str, default="")
    p.add_argument("--trust-remote-code", type=str, default="True")

    # SGLang
    p.add_argument("--sgl-url", type=str, default="http://127.0.0.1:30000")
    p.add_argument("--sgl-api", type=str, default="v1", choices=["v1","native","openai"])
    p.add_argument("--sgl-api-key", type=str, default="")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile-steps", type=int, default=20)
    p.add_argument("--profile-output-dir", type=str, default="./sglang_logs")

    # vllm_openai (vLLM)
    p.add_argument("--vllm_openai-url", type=str, default="http://127.0.0.1:8000/v1")
    p.add_argument("--vllm_openai-api-key", type=str, default="")
    p.add_argument("--vllm_openai-encoding-format", type=str, default="")

    # vllm native (NEW)
    p.add_argument("--vllm-dtype", type=str, default="auto", help="e.g., auto/bfloat16/float16/float32")
    p.add_argument("--vllm-tp", type=int, default=1, help="tensor_parallel_size")
    p.add_argument("--vllm-device", type=str, default="cuda", help="cuda/cpu; will fallback to env VLLM_DEVICE if needed")
    p.add_argument("--vllm-max-model-len", type=int, default=-1, help=">0 to set max_model_len")
    p.add_argument("--vllm-gpu-mem-util", type=float, default=0.90, help="gpu_memory_utilization")

    # MTEB
    p.add_argument("--mteb", action="store_true")
    p.add_argument("--mteb-datasets", type=str, default="")
    p.add_argument("--mteb-langs", type=str, default="")
    p.add_argument("--mteb-task-types", type=str, default="")

    return p.parse_args()

def to_bool(x: str) -> bool:
    return str(x).lower() in ("1","true","yes","on")

def build_encoder(args):
    backend = args.backend
    if backend == "transformers":
        # optional AMX/CPU policy
        try:
            utils.apply_amx_policy(backend="transformers", device=args.device,
                                   use_ipex=to_bool(args.use_ipex), amp=args.amp,
                                   verbose=False)
        except Exception as e:
            print(f"[Warn] apply_amx_policy skipped: {e}")

        return HFEncoder(model=args.model,
                         device=args.device,
                         use_ipex=to_bool(args.use_ipex),
                         amp=args.amp,
                         trust_remote_code=to_bool(args.trust_remote_code),
                         max_length=args.max_length,
                         offline=to_bool(args.offline),
                         hf_endpoint=args.hf_endpoint)

    elif backend == "sglang":
        if SGLangEncoder is None:
            raise RuntimeError("SGLangEncoder not available")
        return SGLangEncoder(base_url=args.sgl_url, model=args.model,
                             api=args.sgl_api, api_key=args.sgl_api_key)

    elif backend == "vllm_openai":
        if VllmOpenAIEncoder is None:
            raise RuntimeError("VllmOpenAIEncoder not available (vllm_openai_backend.py missing?)")
        return VllmOpenAIEncoder(base_url=args.vllm_openai_url,
                                 model=args.model,
                                 api_key=args.vllm_openai_api_key,
                                 encoding_format=(args.vllm_openai_encoding_format or None))

    elif backend == "vllm":
        if VLLMEncoder is None:
            raise RuntimeError("VLLMEncoder not available (vllm_backend.py missing?)")
        # 构造本地 vLLM 的原生 encoder（内部用 LLM(task='embed')）
        kwargs = dict(
            model=args.model,
            dtype=args.vllm_dtype,
            tensor_parallel_size=args.vllm_tp,
            device=args.vllm_device,
            gpu_memory_utilization=args.vllm_gpu_mem_util,
        )
        if args.vllm_max_model_len and args.vllm_max_model_len > 0:
            kwargs["max_model_len"] = args.vllm_max_model_len
        return VLLMEncoder(**kwargs)

    elif backend == "clip":
        return CLIPEncoder(model=args.model, device=args.device,
                           use_ipex=to_bool(args.use_ipex), amp=args.amp,
                           trust_remote_code=to_bool(args.trust_remote_code))
    else:
        raise NotImplementedError(f"Backend '{backend}' is not implemented in this entry.")

# ---------------- Unlabeled Yahoo pipeline (uses datasets.load_yahoo_answers_jsonl) ----------------
def run_unlabeled_yahoo(args, encoder) -> None:
    import torch
    import numpy as np

    if not args.yahoo_jsonl:
        return
    if datasets_mod is None or not hasattr(datasets_mod, "load_yahoo_answers_jsonl"):
        raise RuntimeError("datasets.load_yahoo_answers_jsonl 未找到，请确认 datasets.py 在同目录或 src/ 包内。")

    texts = datasets_mod.load_yahoo_answers_jsonl(
        path=args.yahoo_jsonl,
        mode=args.yahoo_mode,         # 'q' or 'q+a'
        max_records=args.yahoo_max
    )

    if not texts:
        print("[Yahoo] No texts loaded.")
        return
    
    #warm-up
    print(f"[Yahoo] warm-up encoding {min(1000, len(texts))} samples ...")
    warmup_len = min(1000, len(texts))
    _ = encoder.encode(texts[:warmup_len], batch_size=args.batch_size)

    print(f"[Yahoo] total={len(texts)} mode={args.yahoo_mode}, batch_size={args.batch_size}")
    t0 = time.time()
    embs = encoder.encode(texts, batch_size=args.batch_size)
    t1 = time.time()
    dt = t1 - t0
    qps = len(texts) / dt if dt > 0 else float('inf')
    print(f"[Yahoo] time={dt:.3f}s avg_QPS={qps:.2f} shape={tuple(embs.shape)}")

    if args.dump_emb:
        os.makedirs(os.path.dirname(args.dump_emb) or ".", exist_ok=True)
        torch.save(embs.cpu(), args.dump_emb)
        print(f"[Save] embeddings -> {args.dump_emb}")

    if args.output_jsonl:
        rec = {"count": len(texts), "time_sec": dt, "avg_qps": qps,
               "batch_size": args.batch_size, "mode": args.yahoo_mode,
               "model": args.model, "backend": args.backend}
        try:
            utils.append_jsonl(args.output_jsonl, rec)
        except Exception:
            with open(args.output_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[Save] jsonl -> {args.output_jsonl}")

    if args.output_csv:
        row = {"count": len(texts), "time_sec": dt, "avg_qps": qps,
               "batch_size": args.batch_size, "mode": args.yahoo_mode,
               "model": args.model, "backend": args.backend}
        try:
            utils.append_csv(args.output_csv, row, header_if_new=True)
        except Exception:
            import csv
            new = not os.path.exists(args.output_csv)
            with open(args.output_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if new: w.writeheader()
                w.writerow(row)
        print(f"[Save] csv -> {args.output_csv}")

def main():
    args = parse_args()

    # Build encoder
    encoder = build_encoder(args)

    # SGLang server profiling (optional)
    prof_started = False
    if args.backend == "sglang" and args.profile:
        try:
            print("[Profile] start server profiler")
            encoder.start_profile(steps=args.profile_steps, out_dir=args.profile_output_dir)
            prof_started = True
        except Exception as e:
            print(f"[Profile] start failed: {e}")

    # 1) Unlabeled Yahoo JSONL encoding (fast path)
    if args.yahoo_jsonl:
        run_unlabeled_yahoo(args, encoder)

    # 2) Pairwise text classification / Food-101 / generic evals
    ds_list = [s.strip() for s in (args.datasets or "").split(",") if s.strip()]
    if ds_list:
        for ds in ds_list:
            ds_upper = ds.upper()
            if ds_upper == "FOOD101":
                # expect evals.eval_food101_zeroshot(encoder, model_name, split, batch_size, dump_img_emb, dump_txt_emb, output_csv)
                try:
                    evals.eval_food101_zeroshot(
                        encoder=encoder,
                        model_name=args.model,
                        split=args.split,
                        batch_size=args.batch_size,
                        dump_img_emb=args.dump_img_emb,
                        dump_txt_emb=args.dump_txt_emb,
                        output_csv=args.output_csv
                    )
                except Exception as e:
                    print(f"[FOOD101] eval failed: {e}")
            else:
                # generic text-pair datasets
                try:
                    evals.eval_dataset_text_pairs(
                        encoder=encoder,
                        dataset_name=ds,
                        split=args.split,
                        max_samples=(None if args.max_samples<=0 else args.max_samples),
                        batch_size=args.batch_size,
                        output_csv=args.output_csv
                    )
                except Exception as e:
                    print(f"[{ds}] eval failed: {e}")

    # 3) MTEB (optional)
    if args.mteb and mteb is not None:
        task_langs = [s.strip() for s in (args.mteb_langs or "").split(",") if s.strip()] or None
        task_types = [s.strip() for s in (args.mteb_task_types or "").split(",") if s.strip()] or None
        try:
            mteb.run_mteb(
                encoder=encoder,
                model_name=args.model,
                datasets=(args.mteb_datasets or ""),
                batch_size=args.batch_size,
                task_langs=task_langs,
                task_types=task_types
            )
        except Exception as e:
            print(f"[MTEB] run failed: {e}")

    # stop profiling
    if args.backend == "sglang" and args.profile and prof_started:
        try:
            print("[Profile] stop server profiler")
            encoder.stop_profile()
        except Exception as e:
            print(f"[Profile] stop failed: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
