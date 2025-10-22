# 开 AMX
# export CONDA_PREFIX="$(python -c 'import sys,os; print(os.environ.get("CONDA_PREFIX") or os.path.dirname(os.path.dirname(sys.executable)))')"
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

WORK_HOME=$PWD/../
echo "WORK_HOME=$WORK_HOME"

numactl -C 0-16 python $WORK_HOME/main.py \
  --backend llamacpp \
  --model $WORK_HOME/models/bge-large-zh-v1.5 \
  --amx on \
  --llama-lib-amx llama-libs/libllama_amx.so \
  --llama-n-threads 16 \
  --datasets LCQMC,AFQMC --batch-size 16 \
  --output-csv $WORK_HOME/runs/llama_amx_on.csv

# 关 AMX
# numactl -N 0 python embedding_bench.py \
#   --backend llamacpp \
#   --model models/bge-large-zh-v1.5 \
#   --amx off \
#   --llama-lib-noamx llama-libs/libllama_noamx.so \
#   --llama-n-threads 16 \
#   --datasets LCQMC,AFQMC --batch-size 16 \
#   --output-csv runs/llama_amx_off.csv

