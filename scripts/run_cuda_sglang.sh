# unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1,0.0.0.0
export no_proxy=localhost,127.0.0.1,0.0.0.0

# python -m sglang.launch_server \
#   --model-path Qwen/Qwen3-Embedding-4B \
#   --is-embedding --host 0.0.0.0 --port 30000

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

numactl -C 1  python embedding_bench.py \
  --backend sglang \
  --model Qwen/Qwen3-Embedding-4B \
  --sgl-url http://127.0.0.1:30000 \
  --sgl-api  v1 \
  --datasets LCQMC --batch-size 16 \
  --profile
