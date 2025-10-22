#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, argparse
from typing import List, Optional
from src.utils import apply_amx_policy, append_csv, append_jsonl
from src.evals import eval_dataset_text_pairs, eval_food101_zeroshot
from src.mteb_bridge import run_mteb

def parse_args():
    p = argparse.ArgumentParser(description="Embedding eval (LCQMC/AFQMC/PAWSX-zh/FOOD101/MTEB) with AMX control (transformers+IPEX / vLLM / llama.cpp / CLIP / SGLang)")
    p.add_argument("--backend", type=str, default="transformers",
                   choices=["transformers","llamacpp","vllm","clip","sglang"])
    p.add_argument("--model", type=str, default="BAAI/bge-large-zh-v1.5")
    p.add_argument("--datasets", type=str, default="")
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--amx", type=str, default="auto", choices=["auto","on","off"])
    p.add_argument("--amx-verbose", type=str, default="False")

    # transformers / clip common
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--use-ipex", type=str, default="True")
    p.add_argument("--amp", type=str, default="auto", choices=["off","auto","fp16","bf16"])
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--offline", type=str, default="False")
    p.add_argument("--hf-endpoint", type=str, default="")
    p.add_argument("--trust-remote-code", type=str, default="True")

    # llama.cpp
    p.add_argument("--llama-n-threads", type=int, default=0)
    p.add_argument("--llama-n-gpu-layers", type=int, default=0)
    p.add_argument("--llama-verbose", type=str, default="False")
    p.add_argument("--llama-lib-amx", type=str, default="")
    p.add_argument("--llama-lib-noamx", type=str, default="")

    # vLLM
    p.add_argument("--vllm-dtype", type=str, default="auto")
    p.add_argument("--vllm-tp", type=int, default=1)
    p.add_argument("--vllm-device", type=str, default="cpu", choices=["cpu","cuda"])
    p.add_argument("--vllm-max-model-len", type=int, default=8192)
    p.add_argument("--vllm-gpu-mem-util", type=float, default=0.90)

    # CLIP (food101) opts
    p.add_argument("--clip-prompt", type=str, default="a photo of a {}")
    p.add_argument("--dump-img-emb", type=str, default="")
    p.add_argument("--dump-txt-emb", type=str, default="")

    # Yahoo Answers JSONL（无标签编码）
    p.add_argument("--yahoo-jsonl", type=str, default="",
                   help="Path to yahoo_answers_title_answer.jsonl (or similar)")
    p.add_argument("--yahoo-mode", type=str, default="q",
                   choices=["q","q+a"],
                   help="q: only question; q+a: questions and answers (separate rows)")
    p.add_argument("--yahoo-max", type=int, default=-1,
                   help="Only read the first N records from the Yahoo JSONL (default: -1 for all)")
    p.add_argument("--dump-emb", type=str, default="",
                   help="Optional path to dump embeddings for unlabeled text (torch.save)")

    # logging
    p.add_argument("--output-csv", type=str, default="")
    p.add_argument("--output-jsonl", type=str, default="")

    # MTEB
    p.add_argument("--mteb", type=str, default="False")
    p.add_argument("--mteb-tasks", type=str, default="")
    p.add_argument("--mteb-task-langs", type=str, default="")
    p.add_argument("--mteb-task-types", type=str, default="")
    p.add_argument("--mteb-output-dir", type=str, default="runs/mteb")

    # SGLang
    p.add_argument("--sgl-url", type=str, default="http://127.0.0.1:30000")
    p.add_argument("--sgl-api", type=str, default="native", choices=["native","v1","openai"])
    p.add_argument("--sgl-api-key", type=str, default="")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile-steps", type=int, default=20)
    p.add_argument("--profile-output-dir", type=str, default="")

    return p.parse_args()

def main():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    args = parse_args()
    print("[Args]", vars(args))

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    apply_amx_policy(args)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    meta = {"model": args.model, "batch_size": args.batch_size, "timestamp": int(time.time())}

    # backends init
    if args.backend == "transformers":
        from src.backends.transformers_backend import TransformersEncoder
        encoder = TransformersEncoder(
            model_name=args.model,
            device=args.device,
            use_ipex=args.use_ipex,
            amp=args.amp,
            max_length=args.max_length,
            offline=(str(args.offline).lower()=="true"),
            trust_remote_code=(str(args.trust_remote_code).lower()=="true"),
        )
        meta.update({"backend":"transformers","device":args.device,"ipex":args.use_ipex,"amp_mode":args.amp,
                     "onednn_max_cpu_isa": os.environ.get("ONEDNN_MAX_CPU_ISA","<auto>"), "amx": args.amx})

    elif args.backend == "llamacpp":
        from src.backends.llamacpp_backend import LlamaCppEncoder
        encoder = LlamaCppEncoder(
            model_path=args.model,
            n_threads=args.llama_n_threads,
            n_gpu_layers=args.llama_n_gpu_layers,
            verbose=(str(args.llama_verbose).lower()=="true"),
        )
        meta.update({"backend":"llamacpp","threads":args.llama_n_threads,"gpu_layers":args.llama_n_gpu_layers,
                     "llama_lib": os.environ.get("LLAMA_CPP_LIB","<default>"), "amx": args.amx})

    elif args.backend == "vllm":
        from src.backends.vllm_backend import VLLMEncoder
        encoder = VLLMEncoder(
            model=args.model,
            dtype=args.vllm_dtype,
            tensor_parallel_size=args.vllm_tp,
            device=args.vllm_device,
            max_model_len=args.vllm_max_model_len,
            gpu_memory_utilization=args.vllm_gpu_mem_util,
        )
        meta.update({"backend":"vllm","device":args.vllm_device,"dtype":args.vllm_dtype,"tp":args.vllm_tp,
                    "vllm_max_model_len": args.vllm_max_model_len,
                    "vllm_gpu_mem_util": args.vllm_gpu_mem_util,
                    "onednn_max_cpu_isa": os.environ.get("ONEDNN_MAX_CPU_ISA","<auto>"), "amx": args.amx})

    elif args.backend == "clip":
        from src.backends.clip_backend import CLIPEncoder
        encoder = CLIPEncoder(model_name=args.model, device=args.device,
                              use_ipex=args.use_ipex, amp=args.amp)
        meta.update({"backend":"clip","device":args.device,"ipex":args.use_ipex,"amp_mode":args.amp,
                     "onednn_max_cpu_isa": os.environ.get("ONEDNN_MAX_CPU_ISA","<auto>"), "amx": args.amx})

    elif args.backend == "sglang":
        from src.backends.sglang_backend import SGLangEncoder
        encoder = SGLangEncoder(base_url=args.sgl_url, model=args.model,
                                api=args.sgl_api, api_key=args.sgl_api_key)
        meta.update({"backend":"sglang","sgl_url":args.sgl_url,"sgl_api":args.sgl_api})

    else:
        raise ValueError("unknown backend")

    # --- SGLang: profiler 开关 ---
    prof_started = False
    if args.backend == "sglang" and args.profile:
        print(f"[Profile] start server profiler: steps={args.profile_steps}, out_dir='{args.profile_output_dir or '<server-default>'}'")
        prof_started = encoder.start_profile(num_steps=args.profile_steps,
                                             output_dir=args.profile_output_dir)

    # --- Yahoo Answers JSONL 无标签快速路径 ---
    if args.yahoo_jsonl:
        from src.datasets import load_yahoo_answers_jsonl
        from src.evals import eval_unlabeled_texts

        texts = load_yahoo_answers_jsonl(args.yahoo_jsonl, mode=args.yahoo_mode, max_records=args.yahoo_max)
        rec = eval_unlabeled_texts(encoder, texts, batch_size=args.batch_size, dump_emb_path=args.dump_emb)

        if args.output_csv:
            append_csv(args.output_csv, {**rec, "dataset":"yahoo_jsonl", "mode": args.yahoo_mode, "max_records": args.yahoo_max}, meta)
        if args.output_jsonl:
            append_jsonl(args.output_jsonl, {**rec, "dataset":"yahoo_jsonl", "mode": args.yahoo_mode, "max_records": args.yahoo_max}, meta)

        # 如果提供了 --yahoo-jsonl，则跳过下面的成对评测数据集
        datasets = []

    # --- 常规评测（成对/CLIP） ---
    for dk in datasets:
        dk_l = dk.lower()
        if args.backend == "clip" and dk_l == "food101":
            rec = eval_food101_zeroshot(
                encoder, split=args.split, max_samples=args.max_samples,
                batch_size_img=args.batch_size, prompt_template=args.clip_prompt,
                dump_img_emb=args.dump_img_emb, dump_txt_emb=args.dump_txt_emb
            )
        else:
            rec = eval_dataset_text_pairs(
                encoder, dk, split=args.split, batch_size=args.batch_size,
                max_samples=args.max_samples, threshold=args.threshold
            )

        if args.output_csv:   append_csv(args.output_csv, rec, meta)
        if args.output_jsonl: append_jsonl(args.output_jsonl, rec, meta)

    # --- MTEB（可选） ---
    if str(args.mteb).lower() == "true":
        tasks = [t.strip() for t in args.mteb_tasks.split(",") if t.strip()]
        task_langs = [t.strip() for t in args.mteb_task_langs.split(",") if t.strip()] if args.mteb_task_langs else None
        task_types = [t.strip() for t in args.mteb_task_types.split(",") if t.strip()] if args.mteb_task_types else None

        PRESET_ENG_V2 = ["STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark", "SICK-R"]
        PRESET_C_MTEB = ["AFQMC", "LCQMC", "BQ", "PAWSX-zh", "ATEC"]
        PRESET_MULTI  = ["STS17", "STS22"]

        if not tasks:
            if task_langs and ("zh" in task_langs):
                tasks = PRESET_C_MTEB
            elif task_langs and (("mul" in task_langs) or ("multi" in task_langs)):
                tasks = PRESET_MULTI
            else:
                tasks = PRESET_ENG_V2
            print(f"[MTEB] 未指定 --mteb-tasks，使用预置: {tasks}")

        run_mteb(
            encoder=encoder,
            tasks=tasks,
            output_dir=args.mteb_output_dir,
            batch_size=args.batch_size,
            task_langs=task_langs,
            task_types=task_types
        )

    # --- SGLang: 关闭 profiler ---
    if args.backend == "sglang" and args.profile and prof_started:
        print("[Profile] stop server profiler")
        encoder.stop_profile()

    print("\nDone.")

if __name__ == "__main__":
    main()
