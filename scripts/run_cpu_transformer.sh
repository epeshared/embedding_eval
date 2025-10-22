
# export CONDA_PREFIX="$(python -c 'import sys,os; print(os.environ.get("CONDA_PREFIX") or os.path.dirname(os.path.dirname(sys.executable)))')"
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# numactl -C 0-15 python embedding_bench.py \
#   --backend transformers \
#   --model models/bge-large-zh-v1.5 \
#   --offline True \
#   --device cpu --use-ipex True --amp bf16 \
#   --amx on --amx-verbose False \
#   --datasets LCQMC,AFQMC --batch-size 16 \
#   --output-csv runs/transformers_ipex_amx_on.csv


# numactl -C 0-15 python embedding_bench.py \
#   --backend transformers \
#   --model models/bge-large-zh-v1.5 \
#   --offline True \
#   --device cpu --use-ipex True --amp bf16 \
#   --amx off --amx-verbose False \
#   --datasets LCQMC,AFQMC --batch-size 16 \
#   --output-csv runs/transformers_ipex_amx_off.csv


# numactl -C 0-15 python embedding_bench.py \
#   --backend transformers \
#   --model models/bge-large-zh-v1.5 \
#   --offline True \
#   --device cpu --use-ipex True --amp bf16 \
#   --amx on --amx-verbose False \
#   --datasets LCQMC,AFQMC --batch-size 1 \
#   --output-csv runs/transformers_ipex_amx_on.csv


# numactl -C 0-15 python embedding_bench.py \
#   --backend transformers \
#   --model models/bge-large-zh-v1.5 \
#   --offline True \
#   --device cpu --use-ipex True --amp bf16 \
#   --amx off --amx-verbose False \
#   --datasets LCQMC,AFQMC --batch-size 1 \
#   --output-csv runs/transformers_ipex_amx_off.csv


numactl -C 0-7 python embedding_bench.py \
  --backend transformers \
  --model models/Qwen/Qwen3-Embedding-4B \
  --offline True \
  --device cpu --use-ipex True --amp bf16 \
  --amx off --amx-verbose False \
  --datasets LCQMC \
  --batch-size 1 \
  --output-csv runs/transformers_ipex_amx_on.csv