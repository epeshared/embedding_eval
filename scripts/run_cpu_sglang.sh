# unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1,0.0.0.0
export no_proxy=localhost,127.0.0.1,0.0.0.0

WORK_HOME=$PWD/../
echo "WORK_HOME=$WORK_HOME"

# BATCH_SIZE=1
BATCH_SIZE=100

# numactl -C 0-15  python $WORK_HOME/main.py  \
#   --backend sglang \
#   --model $WORK_HOME/models/Qwen/Qwen3-Embedding-0.6B \
#   --sgl-url http://127.0.0.1:30000 \
#   --sgl-api v1 \
#   --amx on --amx-verbose True \
#   --datasets LCQMC --batch-size $BATCH_SIZE \
  # --profile

# numactl -C 0-7  python embedding_bench.py \
#   --backend sglang \
#   --model Qwen/Qwen3-Embedding-4B \
#   --sgl-url http://127.0.0.1:30000 \
#   --sgl-api v1 \
#   --amx on --amx-verbose True \
#   --datasets LCQMC --batch-size 16 \


numactl -C 0-15 python $WORK_HOME/main.py \
  --backend sglang --model $WORK_HOME/models/Qwen/Qwen3-Embedding-4B  \
  --sgl-url http://127.0.0.1:30000 \
  --sgl-api v1 \
  --amx on --amx-verbose False \
  --yahoo-jsonl $WORK_HOME/datasets/yahoo_answers_title_answer.jsonl \
  --yahoo-mode q --yahoo-max 10000 \
  --batch-size $BATCH_SIZE --dump-emb $WORK_HOME/runs/yahoo_q.pt \
  --output-csv $WORK_HOME/runs/yahoo_eval.csv
