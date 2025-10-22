# unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
# export NO_PROXY=localhost,127.0.0.1,0.0.0.0
# export no_proxy=localhost,127.0.0.1,0.0.0.0

#CUDA
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

mkdir -p sglang_logs/sglang_cuda
export SGLANG_TORCH_PROFILER_DIR=sglang_logs/sglang_cuda

WORK_HOME=$PWD/../
echo "WORK_HOME=$WORK_HOME"

BATCH_SIZE=16

# python -m sglang.launch_server \
#   --model-path Qwen/Qwen3-Embedding-4B \
#   --enable-torch-compile \
#   --torch-compile-max-bs 16 \
#   --is-embedding --host 0.0.0.0 --port 30000 --skip-server-warmup --log-level error

numactl -C 0-7 python -m sglang.launch_server \
  --model-path Qwen/Qwen3-Embedding-4B \
  --tokenizer-path Qwen/Qwen3-Embedding-4B \
  --trust-remote-code          \
  --disable-overlap-schedule   \
  --is-embedding \
  --host 0.0.0.0 --port 30000 \
  --skip-server-warmup \
  --log-level error \
  --enable-torch-compile \
  --torch-compile-max-bs $BATCH_SIZE \
