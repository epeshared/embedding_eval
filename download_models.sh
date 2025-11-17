#!/usr/bin/env bash
set -euo pipefail

# 从环境变量读取 HF_TOKEN
: "${HF_TOKEN:?Please set HF_TOKEN env var before running this script}"

export HF_TOKEN

# 后面你自己的下载命令，例如：
# huggingface-cli download xxx/yyy --token "$HF_TOKEN" ...


# huggingface-cli download BAAI/bge-large-zh-v1.5 --local-dir models/bge-large-zh-v1.5 --local-dir-use-symlinks False
# huggingface-cli download openai/clip-vit-base-patch32 --local-dir models/openai/clip-vit-base-patch32 --local-dir-use-symlinks False
huggingface-cli download openai/clip-vit-large-patch14-336 --local-dir models/openai/clip-vit-large-patch14-336 --local-dir-use-symlinks False
# huggingface-cli download --token "$HF_TOKEN"  Qwen/Qwen3-Embedding-4B --local-dir models/Qwen/Qwen3-Embedding-4B
#huggingface-cli download openai/clip-vit-base-patch32 --local-dir models/openai/clip-vit-base-patch32 --local-dir-use-symlinks False
# huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir models/Qwen/Qwen3-Embedding-0.6B --local-dir-use-symlinks False
# huggingface-cli download C-MTEB/LCQMC --local-dir datasets/C-MTEB/LCQMC --local-dir-use-symlinks False
