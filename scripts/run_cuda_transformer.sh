python embedding_bench.py \
  --backend transformers \
  --model models/Qwen/Qwen3-Embedding-4B \
  --device cuda --amp fp16 \
  --datasets LCQMC --batch-size 16
