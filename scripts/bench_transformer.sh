
# export CONDA_PREFIX="$(python -c 'import sys,os; print(os.environ.get("CONDA_PREFIX") or os.path.dirname(os.path.dirname(sys.executable)))')"
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# numactl -C 0-15 python embedding_bench.py \
#   --backend transformers \
#   --model models/bge-large-zh-v1.5 \
#   --offline True \
#   --device cpu --use-ipex True --amp bf16 \
#   --amx on --amx-verbose False \
#   --datasets LCQMC,AFQMC --batch-size 16 \
#   --output-csv runs/transformers_ipex_amx_on.csv


# numactl -C 0-15 python embedding_bench.py \
#   --backend transformers \
#   --model models/bge-large-zh-v1.5 \
#   --offline True \
#   --device cpu --use-ipex True --amp bf16 \
#   --amx off --amx-verbose False \
#   --datasets LCQMC,AFQMC --batch-size 16 \
#   --output-csv runs/transformers_ipex_amx_off.csv


# numactl -C 0-15 python embedding_bench.py \
#   --backend transformers \
#   --model models/bge-large-zh-v1.5 \
#   --offline True \
#   --device cpu --use-ipex True --amp bf16 \
#   --amx on --amx-verbose False \
#   --datasets LCQMC,AFQMC --batch-size 1 \
#   --output-csv runs/transformers_ipex_amx_on.csv


# numactl -C 0-15 python embedding_bench.py \
#   --backend transformers \
#   --model models/bge-large-zh-v1.5 \
#   --offline True \
#   --device cpu --use-ipex True --amp bf16 \
#   --amx off --amx-verbose False \
#   --datasets LCQMC,AFQMC --batch-size 1 \
#   --output-csv runs/transformers_ipex_amx_off.csv


numactl -C 0-7 python embedding_bench.py \
  --backend transformers \
  --model models/Qwen/Qwen3-Embedding-4B \
  --offline True \
  --device cpu --use-ipex True --amp bf16 \
  --amx off --amx-verbose False \
  --datasets LCQMC \
  --batch-size 1 \
  --output-csv runs/transformers_ipex_amx_on.csv

#!/bin/bash

# 你想测试的 Batch Size 列表
# SIZES=(1 2 4 8 16 32 64 100 128)

# for BATCH_SIZE in "${SIZES[@]}"; do
#     echo "=============================================="
#     echo "Running with BATCH_SIZE = $BATCH_SIZE"
#     echo "=============================================="

#     python ../main.py \
#       --backend transformers \
#       --model ../models/Qwen/Qwen3-Embedding-4B \
#       --offline True \
#       --device cpu --use-ipex True --amp bf16 \
#       --amx on --amx-verbose False \
#       --yahoo-jsonl $WORK_HOME/datasets/yahoo_answers_title_answer.jsonl \
#       --yahoo-mode q \
#       --yahoo-max 1000 \
#       --batch-size $BATCH_SIZE \
#       --output-csv runs/transformers_ipex_amx_on_bs${BATCH_SIZE}.csv
# done
