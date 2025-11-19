#!/usr/bin/env bash
set -euo pipefail

WORK_HOME=$PWD/../
echo "WORK_HOME=$WORK_HOME"

###############################################
#        ✅ 仅需在这里配置模型路径即可
###############################################
# MODEL_DIR="$WORK_HOME/models/openai/clip-vit-base-patch32"
# MODEL_DIR="$WORK_HOME/models/openai/clip-vit-large-patch14-336"
MODEL_DIR="$WORK_HOME/models/Qwen/Qwen3-Embedding-4B"
# MODEL_DIR="$WORK_HOME/models/Qwen/Qwen3-Embedding-0.6B"
###############################################
echo "Using model: $MODEL_DIR"

# GPU
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0


# ===== BATCH_SIZE 列表 =====
# BATCH_LIST=(1 2 4 8 16 32 64 100 128)
BATCH_LIST=(100)

for BATCH_SIZE in "${BATCH_LIST[@]}"; do
    echo "=============================="
    echo "Running BATCH_SIZE=$BATCH_SIZE"
    echo "=============================="

    numactl -C 0-7,256-263 \
    python $WORK_HOME/main.py \
      --backend sglang-offline \
      --model "$MODEL_DIR" \
      --device cuda \
      --yahoo-jsonl $WORK_HOME/datasets/yahoo_answers_title_answer.jsonl \
      --yahoo-mode q \
      --yahoo-max 10000 \
      --batch-size $BATCH_SIZE \
      --dump-emb $WORK_HOME/runs/yahoo_q_bs${BATCH_SIZE}.pt \
      --output-csv $WORK_HOME/runs/yahoo_eval_bs${BATCH_SIZE}.csv     

done
echo "All done."

# python $WORK_HOME/src/backends/sglang_offline_backend.py --model-path "$MODEL_DIR" --device cpu