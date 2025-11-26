# 你想测试的 Batch Size 列表
#SIZES=(1 2 4 8 16 32 64 100 128)
SIZES=(100)

#MODEL="/home/xtang/models/Qwen/Qwen3-Embedding-0.6B"
MODEL="/home/xtang/models/BAAI/bge-large-zh-v1.5"

for BATCH_SIZE in "${SIZES[@]}"; do
     echo "=============================================="
     echo "Running with BATCH_SIZE = $BATCH_SIZE"
     echo "=============================================="

     python ../main.py \
       --backend transformers \
       --model $MODEL \
       --offline True \
       --device cpu --use-ipex False --amp bf16 \
       --yahoo-jsonl /home/xtang/datasets/yahoo_answers_title_answer.jsonl \
       --yahoo-mode q \
       --yahoo-max 10000 \
       --batch-size $BATCH_SIZE
done
