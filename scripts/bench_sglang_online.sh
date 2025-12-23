#!/bin/bash
set -euo pipefail

export NO_PROXY=localhost,127.0.0.1,0.0.0.0
export no_proxy=localhost,127.0.0.1,0.0.0.0

# project root: assume this script is in <repo>/scripts/...
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_HOME="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "WORK_HOME=$WORK_HOME"

# ===== 手动配置模型 =====
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-Embedding-4B}"
echo "Using model: $MODEL_PATH"

# ===== BATCH_SIZE 列表 =====
BATCH_LIST=(10)

# ===== 每个 batch_size 启动多少个 main.py 进程 =====
NUM_RUNS_PER_BS="${NUM_RUNS_PER_BS:-1}"

LOG_DIR="${LOG_DIR:-$WORK_HOME/logs}"
RUN_DIR="${RUN_DIR:-$WORK_HOME/runs}"
OUT_DIR="${OUT_DIR:-$WORK_HOME/outputs}"

mkdir -p "$LOG_DIR" "$RUN_DIR" "$OUT_DIR"

# ===== profile 开关（可配置）=====
# 用法：
#   PROFILE=1 ./bench.sh    # 开
#   PROFILE=0 ./bench.sh    # 关
PROFILE="${PROFILE:-0}"

# 组装 profile 参数（PROFILE=1 才加 --profile）
PROFILE_ARGS=()
if [[ "$PROFILE" == "1" || "$PROFILE" == "true" || "$PROFILE" == "True" ]]; then
  PROFILE_ARGS+=(--profile)
fi

for BATCH_SIZE in "${BATCH_LIST[@]}"; do
  echo "=============================="
  echo "Running BATCH_SIZE=$BATCH_SIZE"
  echo "=============================="

  for RUN_ID in $(seq 1 "$NUM_RUNS_PER_BS"); do
    LOG_FILE="$LOG_DIR/bs${BATCH_SIZE}_run${RUN_ID}.log"
    OUT_CSV="$RUN_DIR/yahoo_eval_bs${BATCH_SIZE}_run${RUN_ID}.csv"

    # 每个 run 单独输出，避免覆盖/并发写冲突
    DUMP_JSONL="$OUT_DIR/yahoo_emb_bs${BATCH_SIZE}_run${RUN_ID}.jsonl"

    echo "Start: BATCH_SIZE=$BATCH_SIZE, RUN_ID=$RUN_ID => $LOG_FILE"

    python "$WORK_HOME/main.py" \
      --backend sglang-online \
      --model "$MODEL_PATH" \
      --sgl-url http://127.0.0.1:30000 \
      --sgl-api v1 \
      --yahoo-jsonl /home/xtang/datasets/yahoo_answers_title_answer.jsonl \
      --yahoo-mode q \
      --yahoo-max 10 \
      --batch-size "$BATCH_SIZE" \
      --dump-emb-jsonl "$DUMP_JSONL" \
      --dump-emb-with-text True \
      --output-csv "$OUT_CSV" \
      "${PROFILE_ARGS[@]}" \
      >"$LOG_FILE" 2>&1 &

    echo "Started PID $! for BATCH_SIZE=$BATCH_SIZE / RUN_ID=$RUN_ID"
  done
done

echo "All jobs started. Waiting for them to finish..."
wait
echo "All done."
