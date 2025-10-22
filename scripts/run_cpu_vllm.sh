# amx 开

# export HF_ENDPOINT="https://hf-mirror.com"
# export HF_HUB_BASE_URL="https://hf-mirror.com"

# export CONDA_PREFIX="$(python -c 'import sys,os; print(os.environ.get("CONDA_PREFIX") or os.path.dirname(os.path.dirname(sys.executable)))')"
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

numactl -C 0-7 python embedding_bench.py \
  --backend vllm \
  --model openai/clip-vit-base-patch32  \
  --vllm-device cpu \
  --vllm-dtype bfloat16 \
  --amx on --amx-verbose True \
  --datasets LCQMC,AFQMC --batch-size 16 \
  --output-csv runs/vllm_amx_on.csv


# amx 关
# numactl -N 0 python embedding_bench.py \
#   --backend vllm \
#   --model models/bge-large-zh-v1.5  \
#   --vllm-device cpu \
#   --vllm-dtype bfloat16 \
#   --amx off --amx-verbose True \
#   --datasets LCQMC,AFQMC --batch-size 16 \
#   --output-csv runs/vllm_amx_off.csv

