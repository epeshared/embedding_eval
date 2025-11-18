#!/bin/bash

# unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1,0.0.0.0
export no_proxy=localhost,127.0.0.1,0.0.0.0

WORK_HOME=$PWD/../
echo "WORK_HOME=$WORK_HOME"

# ===== 手动配置模型 =====
# 默认模型（你可以外部 export MODEL_PATH 覆盖）
# MODEL_PATH="Qwen/Qwen3-Embedding-0.6B"
MODEL_PATH="Qwen/Qwen3-Embedding-4B"
echo "Using model: $MODEL_PATH"

# ===== BATCH_SIZE 列表 =====
# BATCH_LIST=(1 2 4 8 16 32 64 100 128)
BATCH_LIST=(100)

for BATCH_SIZE in "${BATCH_LIST[@]}"; do
    echo "=============================="
    echo "Running BATCH_SIZE=$BATCH_SIZE"
    echo "=============================="

    numactl -C 0-63 python $WORK_HOME/main.py \
      --backend sglang-online  \
      --model "$MODEL_PATH" \
      --sgl-url http://127.0.0.1:30000 \
      --sgl-api v1 \
      --yahoo-jsonl $WORK_HOME/datasets/yahoo_answers_title_answer.jsonl \
      --yahoo-mode q \
      --yahoo-max 10000 \
      --batch-size $BATCH_SIZE \
      --dump-emb $WORK_HOME/runs/yahoo_q_bs${BATCH_SIZE}.pt \
      --output-csv $WORK_HOME/runs/yahoo_eval_bs${BATCH_SIZE}.csv  

done
echo "All done."