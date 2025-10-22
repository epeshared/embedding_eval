
# emb_eval — Modular Embedding Evaluation Suite

A refactor of your original single-file script into a maintainable, **modular** package with a clean CLI, per‑backend implementations, and handy utilities for:
- Text pair similarity evaluation (LCQMC / AFQMC / PAWSX‑zh + local/custom datasets)
- Zero‑shot **CLIP** evaluation on Food‑101
- **MTEB** one‑click evaluation via an adapter
- **AMX** policy toggles (auto/on/off) for CPU backends (oneDNN/IPEX, llama.cpp variants)
- Remote **SGLang** server usage (native `/encode`, OpenAI‑compatible `/v1/embeddings`, or `openai` SDK)
- **vLLM** (CPU/GPU) local Python API for embedding

> The code is split per backend to keep things small, focused, and easy to extend.

---

## Contents

- [Layout](#layout)
- [Installation](#installation)
- [Quickstart](#quickstart)
  - [Transformers (CPU + IPEX/AMX)](#transformers-cpu--ipexamx)
  - [vLLM (GPU or CPU)](#vllm-gpu-or-cpu)
  - [llama.cpp (GGUF)](#llamacpp-gguf)
  - [SGLang server](#sglang-server)
  - [CLIP zero‑shot (Food‑101)](#clip-zero-shot-food-101)
- [CLI Arguments](#cli-arguments)
- [Datasets (remote + local)](#datasets-remote--local)
- [AMX Policy & oneDNN/IPEX](#amx-policy--onednnipex)
- [MTEB Integration](#mteb-integration)
- [Outputs (CSV, JSONL)](#outputs-csv-jsonl)
- [Profiling SGLang](#profiling-sglang)
- [Troubleshooting](#troubleshooting)
- [Tips & Performance Notes](#tips--performance-notes)
- [Extend with a New Backend](#extend-with-a-new-backend)
- [License](#license)

---

## Layout

```
embedding/
  main.py                      # CLI entry (argument names kept compatible with your original script)
  src/
    __init__.py
    utils.py                   # mean_pooling, l2_normalize, CSV/JSONL helpers, AMX policy
    datasets.py                # Remote & local dataset loaders (auto column detection, recursive file search)
    evals.py                   # Text-pair evaluation + Food-101 zero-shot
    mteb_bridge.py             # MTEB adapter wrapper
    backends/
      __init__.py
      base.py                  # BaseEncoder interface
      transformers_backend.py  # TransformersEncoder (IPEX/AMP/CPU/GPU)
      vllm_backend.py          # VLLMEncoder (embed task)
      llamacpp_backend.py      # LlamaCppEncoder (gguf)
      sglang_backend.py        # SGLangEncoder (native/v1/openai; batch & non-batch)
      clip_backend.py          # CLIPEncoder (text/image features, supports IPEX)
```

---

## Installation

Python **3.10+** recommended.

Base dependencies (common path):
```bash
pip install torch datasets scikit-learn transformers pillow requests
```

Optional (install as needed by your chosen backend):
```bash
# vLLM backend
pip install vllm

# llama.cpp backend
pip install llama-cpp-python

# MTEB (only if you run --mteb True)
pip install mteb

# OpenAI-compatible SDK (only if you use --sgl-api openai)
pip install openai

# Intel IPEX (CPU acceleration via oneDNN/AMX; optional)
pip install intel-extension-for-pytorch
```

> If you use mirrors (e.g., HF), you can set `HF_ENDPOINT` (see below).

---

## Quickstart

All commands are run from `emb_eval_refactor/`:

### Transformers (CPU + IPEX/AMX)

```bash
python main.py --backend transformers --model BAAI/bge-large-zh-v1.5 \
  --device cpu --use-ipex True --amp bf16 --amx on \
  --datasets LCQMC --batch-size 16 \
  --output-csv runs/tf_ipex.csv
```

- `--amx on` forces `ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX` (via IPEX).
- Use `--offline True` to avoid network calls (requires models present locally).
- Use `--hf-endpoint https://huggingface.co` (or your mirror) if needed.

### vLLM (GPU or CPU)

```bash
# GPU (single GPU)
CUDA_VISIBLE_DEVICES=0 python main.py --backend vllm --model Qwen/Qwen3-Embedding-4B \
  --vllm-device cuda --vllm-dtype auto \
  --datasets LCQMC --batch-size 16

# CPU
python main.py --backend vllm --model intfloat/e5-mistral-7b-instruct \
  --vllm-device cpu --vllm-dtype bfloat16 --amx on \
  --datasets LCQMC --batch-size 16
```

### llama.cpp (GGUF)

```bash
python main.py --backend llamacpp --model models/bge-large-zh-v1.5.gguf \
  --llama-n-threads 16 --datasets LCQMC --batch-size 32

# Select AMX or non‑AMX builds
python main.py --backend llamacpp --model models/xxx.gguf \
  --amx on  --llama-lib-amx   /path/to/libllama_amx.so
python main.py --backend llamacpp --model models/xxx.gguf \
  --amx off --llama-lib-noamx /path/to/libllama_noamx.so
```

### SGLang server

Server should be started separately (CPU/GPU/AMX decided on server side).

```bash
# v1 (OpenAI-compatible) — supports batch & non-batch
python main.py --backend sglang --model Qwen/Qwen3-Embedding-4B \
  --sgl-url http://127.0.0.1:30000 --sgl-api v1 \
  --datasets LCQMC --batch-size 32

# native (/encode) — robust single-text endpoint
python main.py --backend sglang --model Qwen/Qwen3-Embedding-4B \
  --sgl-url http://127.0.0.1:30000 --sgl-api native \
  --datasets LCQMC --batch-size 32

# via OpenAI SDK — if the server exposes /v1
python main.py --backend sglang --model Qwen/Qwen3-Embedding-4B \
  --sgl-url http://127.0.0.1:30000 --sgl-api openai --sgl-api-key sk-xxx \
  --datasets LCQMC
```

### CLIP zero‑shot (Food‑101)

```bash
python main.py --backend clip --model openai/clip-vit-base-patch32 \
  --datasets food101 --batch-size 64 \
  --clip-prompt "a photo of a {}"
```

Dump embeddings (optional):
```bash
... --dump-txt-emb runs/clip_text_emb.pt --dump-img-emb runs/clip_img_emb.pt
```

---

## CLI Arguments

**Backend selection**
- `--backend`: `transformers` | `llamacpp` | `vllm` | `clip` | `sglang`

**Common**
- `--model`: model ID or path (varies per backend)
- `--datasets`: comma‑separated list (`LCQMC,AFQMC,pawsx-zh` or local path); leave empty to **skip local eval** and only run `--mteb True` if you want
- `--split`: dataset split (default `validation`)
- `--max-samples`: cap evaluation samples (`-1`=all)
- `--batch-size`: batch for encoding
- `--threshold`: cosine threshold for binary similarity (text-pair tasks)

**AMX (CPU)**
- `--amx`: `auto|on|off`
- `--amx-verbose`: `True|False` (sets `ONEDNN_VERBOSE=1`)

**Transformers/CLIP**
- `--device`: `cpu|cuda`
- `--use-ipex`: `True|False` (CPU only)
- `--amp`: `off|auto|fp16|bf16`
- `--max-length`: tokenizer max length
- `--offline`: `True|False` (HuggingFace local-only)
- `--hf-endpoint`: override `HF_ENDPOINT` (e.g., mirror)
- `--trust-remote-code`: `True|False`

**llama.cpp**
- `--llama-n-threads`, `--llama-n-gpu-layers`, `--llama-verbose`
- `--llama-lib-amx`, `--llama-lib-noamx` to select a specific shared library

**vLLM**
- `--vllm-dtype`: `auto|float32|bfloat16|float16|...`
- `--vllm-tp`: tensor parallel size
- `--vllm-device`: `cpu|cuda`
- `--vllm-max-model-len`: cap KV cache size for embed
- `--vllm-gpu-mem-util`: default `0.90`

**SGLang**
- `--sgl-url`: server base URL (e.g., `http://127.0.0.1:30000`)
- `--sgl-api`: `native|v1|openai`
- `--sgl-api-key`: Bearer token (if your server requires it)
- `--profile`: start server profiler before eval
- `--profile-steps`: number of forward steps to capture
- `--profile-output-dir`: requested output dir on server (server-dependent)

**Logging**
- `--output-csv`, `--output-jsonl`

**MTEB**
- `--mteb`: `True|False`
- `--mteb-tasks`: comma‑separated task list; if empty, auto‑selects presets
- `--mteb-task-langs`, `--mteb-task-types`
- `--mteb-output-dir`: output folder for MTEB results

---

## Datasets (remote & local)

Remote datasets supported out‑of‑the‑box:
- `LCQMC` → `C-MTEB/LCQMC`
- `AFQMC` → `clue/afqmc`
- `pawsx-zh` | `pawsx` | `paws-x-zh` → `paws-x` (zh)

Local datasets:
- Pass a **directory**: it will try `load_dataset(dir)` → `load_from_disk(dir)` → recursive search for files.
- Pass a **single file**: supports `csv/tsv/json/jsonl/parquet`.

Auto column detection (case‑insensitive):
- **Pairs**: one of `sentence1/sentence2`, `text1/text2`, `question1/question2`, `q/p`, `s1/s2`, `query/passage`
- **Label**: one of `label`, `score`, `target`, `y`, `similar`  
  - If numeric → `>=0.5` considered positive; if boolean → cast to 0/1; if string → accepts `1/0/true/false/yes/no/y/n`

Specify `--split` (default `validation`). For directory datasets, we try common aliases: `validation|valid|dev|test|train`.

---

## AMX Policy & oneDNN/IPEX

`--amx` affects CPU paths:

- `transformers` / `clip` with `--device cpu` and `--use-ipex True`:
  - `on`  → `ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX`
  - `off` → `ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16`
  - `auto` → let oneDNN pick best ISA
  - `--amx-verbose True` → `ONEDNN_VERBOSE=1`

- `vllm` (CPU builds): sets the same oneDNN env var for vLLM’s CPU path.

- `llamacpp`:
  - `--amx on`  + `--llama-lib-amx=/path/to/libllama_amx.so`
  - `--amx off` + `--llama-lib-noamx=/path/to/libllama_noamx.so`

> AMX actually requires compatible CPUs (e.g., Intel SPR/GNR) and kernel/OS support.

---

## MTEB Integration

Run a curated subset quickly:
```bash
python main.py --backend transformers --model BAAI/bge-large-zh-v1.5 \
  --device cpu --use-ipex True --amp bf16 --amx on \
  --mteb True --mteb-task-langs zh --mteb-output-dir runs/mteb
```
If `--mteb-tasks` is empty, presets are chosen by `--mteb-task-langs`/`--mteb-task-types`:
- zh: `["AFQMC","LCQMC","BQ","PAWSX-zh","ATEC"]`
- multi: `["STS17","STS22"]`
- default English STS‑V2 set: `["STS12","STS13","STS14","STS15","STS16","STSBenchmark","SICK-R"]`

> Requires `pip install mteb`.

---

## Outputs (CSV, JSONL)

Each evaluated dataset produces a record like:
```json
{
  "dataset": "LCQMC",
  "split": "validation",
  "n_samples": 12345,
  "acc": 0.8621,
  "f1": 0.8590,
  "encode_time": 12.345678,
  "score_time": 0.012345,
  "total_time": 12.358023,
  "qps": 999.99
  // plus metadata columns merged from CLI context
}
```

- `--output-csv runs/xxx.csv` → appends a CSV row (headers written if the file is new)
- `--output-jsonl runs/xxx.jsonl` → appends compact JSON lines

---

## Profiling SGLang

If your SGLang server implements `/start_profile` and `/stop_profile`, you can wrap a run:
```bash
python main.py --backend sglang --model Qwen/Qwen3-Embedding-4B \
  --sgl-url http://127.0.0.1:30000 --sgl-api v1 \
  --profile --profile-steps 20 --profile-output-dir /tmp/sgl-trace \
  --datasets LCQMC
```
The client will request server-side Torch Profiler to start and stop around the evaluation.

---

## Troubleshooting

- **Transformers can’t download model**  
  Set a mirror or go offline:
  ```bash
  export HF_ENDPOINT=https://huggingface.co   # or your mirror
  python main.py ... --offline True
  ```

- **vLLM TypeError about `device` argument**  
  Some versions removed `device=`; we transparently fall back to `VLLM_DEVICE` env.

- **SGLang `/v1/embeddings` errors**  
  Ensure the server actually exposes `/v1/embeddings` and your `--sgl-api` matches.
  If auth is enabled, provide `--sgl-api-key` (Bearer token).

- **AMX not taking effect**  
  - Verify CPU supports AMX.
  - For Transformers/CLIP: `--device cpu --use-ipex True --amp bf16 --amx on`
  - Optionally set `--amx-verbose True` to print oneDNN kernel choices.

- **Local dataset columns not found**  
  Check your headers; supported aliases are listed above. For JSON/JSONL, ensure fields exist at the top level.

- **CLIP missing PIL**  
  `pip install pillow`

---

## Tips & Performance Notes

- Pin CPU cores with `numactl` for consistent results:
  ```bash
  numactl -C 0-31 -m 0 python main.py ...
  ```
- On AMX‑capable CPUs, prefer BF16 (`--amp bf16`) with IPEX to enable oneDNN fused kernels.
- For GPUs, manage device selection via `CUDA_VISIBLE_DEVICES` (vLLM) or server process (SGLang).

---

## Extend with a New Backend

1. Create `emb_eval/backends/my_backend.py` implementing a class with:
   ```python
   class MyEncoder:
       @torch.inference_mode()
       def encode(self, texts: List[str], batch_size: int = 128) -> torch.Tensor:
           ...
   ```
2. Import it from `backends/__init__.py` (optional).
3. Add an `elif` block in `main.py` to instantiate it from CLI flags.

---

## License

This repository contains evaluation utilities and wrappers. You are responsible for complying with the licenses of any third‑party models, datasets, or libraries you use.

---

## Acknowledgements

- Hugging Face `transformers`, `datasets`
- Intel `intel-extension-for-pytorch`
- vLLM
- llama.cpp / llama-cpp-python
- SGLang
- MTEB
