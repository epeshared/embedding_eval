# Embedding Evaluation Suite

A modular toolkit for evaluating embedding models, supporting multiple backends and tasks. The structure is clean, extensible, and easy to maintain.

## Directory Structure

```
embedding_eval/
│
├── main.py                # CLI entry, compatible with the original script
├── src/
│   ├── __init__.py
│   ├── utils.py           # pooling, normalization, CSV/JSONL helpers, AMX policy
│   ├── datasets.py        # dataset loader (remote/local/auto column detection)
│   ├── evals.py           # text-pair evaluation, Food-101 zero-shot
│   ├── mteb_bridge.py     # MTEB adapter
│   └── backends/
│       ├── __init__.py
│       ├── base.py
│       ├── transformers_backend.py
│       ├── vllm_backend.py
│       ├── llamacpp_backend.py
│       ├── sglang_backend.py
│       ├── clip_backend.py
│       └── vllm_openai_backend.py
├── scripts/
│   ├── bench_sglang.sh
│   ├── bench_transformer.sh
│   ├── bench_vllm_cuda.sh
│   ├── bench_vllm_openai.sh
│   ├── start_sglang_server.sh
│   ├── start_vllm_server.sh
│   └── runs/
│       └── ...
├── requirements-cpu.txt
├── requirements-cuda.txt
├── requirements-ipex.txt
├── README.md
└── ...  # other files and folders
```

## Installation

Python 3.10+ is recommended.

Basic dependencies (CPU/general):
```bash
pip install -r requirements-cpu.txt
```
For GPU, IPEX, vLLM, etc., install the corresponding requirements file or install manually:
```bash
pip install -r requirements-cuda.txt
pip install -r requirements-ipex.txt
```

Some backends/features require extra dependencies:
- vLLM: `pip install vllm`
- llama.cpp: `pip install llama-cpp-python`
- MTEB: `pip install mteb`
- OpenAI SDK: `pip install openai`
- IPEX: `pip install intel-extension-for-pytorch`
- CLIP: `pip install pillow`

## Quick Start

Run all commands from the project root directory.

### Transformers (CPU/IPEX/AMX)

```bash
python main.py --backend transformers --model BAAI/bge-large-zh-v1.5 \
  --device cpu --use-ipex True --amp bf16 --amx on \
  --datasets LCQMC --batch-size 16 \
  --output-csv runs/tf_ipex.csv
```

### vLLM (GPU/CPU)

```bash
# GPU
CUDA_VISIBLE_DEVICES=0 python main.py --backend vllm --model Qwen/Qwen3-Embedding-4B \
  --vllm-device cuda --vllm-dtype auto --datasets LCQMC --batch-size 16

# CPU
python main.py --backend vllm --model intfloat/e5-mistral-7b-instruct \
  --vllm-device cpu --vllm-dtype bfloat16 --amx on \
  --datasets LCQMC --batch-size 16
```

### llama.cpp (GGUF)

```bash
python main.py --backend llamacpp --model models/bge-large-zh-v1.5.gguf \
  --llama-n-threads 16 --datasets LCQMC --batch-size 32
```

### SGLang Server

Start the server separately (see `scripts/start_sglang_server.sh`), then run:

```bash
python main.py --backend sglang --model Qwen/Qwen3-Embedding-4B \
  --sgl-url http://127.0.0.1:30000 --sgl-api v1 \
  --datasets LCQMC --batch-size 32
```

### CLIP zero-shot (Food-101)

```bash
python main.py --backend clip --model openai/clip-vit-base-patch32 \
  --datasets food101 --batch-size 64 \
  --clip-prompt "a photo of a {}"
```

## Script Tools

The `scripts/` directory contains common benchmark and server startup scripts:

- `bench_transformer.sh`: Transformers backend benchmark
- `bench_vllm_cuda.sh`: vLLM CUDA benchmark
- `bench_vllm_openai.sh`: vLLM OpenAI interface benchmark
- `bench_sglang.sh`: SGLang benchmark
- `start_sglang_server.sh`: Start SGLang server
- `start_vllm_server.sh`: Start vLLM server

You can modify script parameters for batch testing.

## Common Arguments

- `--backend`: Backend type (transformers/llamacpp/vllm/clip/sglang)
- `--model`: Model ID or path
- `--datasets`: Dataset name or local path, supports multiple formats
- `--batch-size`: Batch size
- `--amx`: AMX policy (auto/on/off)
- `--output-csv`/`--output-jsonl`: Output result path
- More arguments: see `main.py` and backend implementations

## Dataset Support

- Remote: LCQMC, AFQMC, PAWSX-zh, etc.
- Local: Supports csv/tsv/json/jsonl/parquet, auto column detection
- Directory: Recursively searches for common data files

## Evaluation Output

Each dataset evaluation produces a record, supporting CSV/JSONL format, including accuracy, F1, QPS, timing, and more.

## Extend a New Backend

1. Create `src/backends/xxx_backend.py` and implement the `encode` method.
2. Register the new backend in `main.py`.
3. Add scripts and documentation if needed.

