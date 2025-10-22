python embedding_bench.py \
  --backend llamacpp \
  --model /path/to/model.gguf \
  --llama-n-gpu-layers 40 \
  --datasets LCQMC,AFQMC --batch-size 16
