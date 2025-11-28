#!/bin/bash

export NO_PROXY=localhost,127.0.0.1,0.0.0.0
export no_proxy=localhost,127.0.0.1,0.0.0.0

WORK_HOME=$PWD/../
echo "WORK_HOME=$WORK_HOME"

# ===== 手动配置模型 =====
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-Embedding-4B}"
echo "Using model: $MODEL_PATH"

# ===== BATCH_SIZE 列表 =====
# BATCH_LIST=(1 2 4 8 16 32 64 100 128)
BATCH_LIST=(100)

# ===== 每个 batch_size 启动多少个 main.py 进程 =====
NUM_RUNS_PER_BS=1      # 这里改数字即可，例如 4 表示启动 4 个 main.py

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

for BATCH_SIZE in "${BATCH_LIST[@]}"; do
    echo "=============================="
    echo "Running BATCH_SIZE=$BATCH_SIZE"
    echo "=============================="

    for RUN_ID in $(seq 1 $NUM_RUNS_PER_BS); do
        LOG_FILE="$LOG_DIR/bs${BATCH_SIZE}_run${RUN_ID}.log"
        DUMP_EMB="$WORK_HOME/runs/yahoo_q_bs${BATCH_SIZE}_run${RUN_ID}.pt"
        OUT_CSV="$WORK_HOME/runs/yahoo_eval_bs${BATCH_SIZE}_run${RUN_ID}.csv"

        echo "Start: BATCH_SIZE=$BATCH_SIZE, RUN_ID=$RUN_ID => $LOG_FILE"

        python "$WORK_HOME/main.py" \
          --backend sglang-online \
          --model "$MODEL_PATH" \
          --sgl-url http://127.0.0.1:30000 \
          --sgl-api v1 \
          --yahoo-jsonl $WORK_HOME/datasets/yahoo_answers_title_answer.jsonl\
          --yahoo-mode q \
          --yahoo-max 1000 \
          --batch-size "$BATCH_SIZE" \
          --dump-emb "$DUMP_EMB" \
          --output-csv "$OUT_CSV" \
          --profile \
          >"$LOG_FILE" 2>&1 &

        echo "Started PID $! for BATCH_SIZE=$BATCH_SIZE / RUN_ID=$RUN_ID"
    done
done

echo "All jobs started. Waiting for them to finish..."
wait
echo "All done."
