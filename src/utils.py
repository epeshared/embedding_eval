
import os, csv, json, time
from typing import Dict, Optional
import torch
import torch.nn.functional as F

def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom  = mask.sum(dim=1).clamp_min(1e-9)
    return summed / denom

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / torch.clamp(x.norm(p=2, dim=dim, keepdim=True), min=eps)

def batched_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=1)

def append_csv(path: str, record: Dict, extra: Dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    row = {**record, **extra}; header = list(row.keys())
    write_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header: w.writeheader()
        w.writerow(row)

def append_jsonl(path: str, record: Dict, extra: Dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({**record, **extra}, ensure_ascii=False) + "\n")

# ========= AMX policy =========
def apply_amx_policy(args):
    amx = args.amx.lower()
    if amx not in ("auto", "on", "off"):
        raise ValueError("--amx must be auto|on|off")

    def set_onednn_env(tag: str):
        if args.amx == "on":
            os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX512_CORE_AMX"
        elif args.amx == "off":
            os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX512_CORE_BF16"
        if str(args.amx_verbose).lower() == "true":
            os.environ["ONEDNN_VERBOSE"] = "1"
        print(f"[AMX:{tag}] ONEDNN_MAX_CPU_ISA={os.environ.get('ONEDNN_MAX_CPU_ISA','<auto>')} "
              f"ONEDNN_VERBOSE={os.environ.get('ONEDNN_VERBOSE','0')}")

    backend = args.backend.lower()

    if backend in ("transformers", "clip"):
        use_ipex = (str(args.use_ipex).lower() == "true")
        if "cpu" in args.device.lower() and use_ipex:
            set_onednn_env(backend)
        else:
            print(f"[AMX:{backend}] Skipped (only applies on CPU + --use-ipex True).")

    if backend == "vllm":
        set_onednn_env("vllm")

    if backend == "llamacpp":
        if amx == "on" and args.llama_lib_amx:
            os.environ["LLAMA_CPP_LIB"] = args.llama_lib_amx
            print(f"[AMX:llamacpp] Using AMX build: {args.llama_lib_amx}")
        elif amx == "off" and args.llama_lib_noamx:
            os.environ["LLAMA_CPP_LIB"] = args.llama_lib_noamx
            print(f"[AMX:llamacpp] Using NO-AMX build: {args.llama_lib_noamx}")
        else:
            print("[AMX:llamacpp] No explicit lib provided; default library will be used.")
