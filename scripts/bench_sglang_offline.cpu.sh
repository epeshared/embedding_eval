#!/usr/bin/env bash
set -euo pipefail

WORK_HOME=$PWD/../
echo "WORK_HOME=$WORK_HOME"

###############################################
#        ✅ 仅需在这里配置模型路径即可
###############################################
# MODEL_DIR="$WORK_HOME/models/openai/clip-vit-base-patch32"
# MODEL_DIR="$WORK_HOME/models/openai/clip-vit-large-patch14-336"
MODEL_DIR="$WORK_HOME/models/Qwen/Qwen3-Embedding-4B"
# MODEL_DIR="$WORK_HOME/models/Qwen/Qwen3-Embedding-0.6B"
###############################################
echo "Using model: $MODEL_DIR"

# ===== OneDNN / IPEX 建议 =====
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
export DNNL_VERBOSE=0
export IPEX_DISABLE_AUTOCAST=1   # 建议开启，规避 uint64 copy_kernel 坑

# ===== 日志目录 =====
# mkdir -p "sglang_logs/sglang_cpu"
# export SGLANG_TORCH_PROFILER_DIR="$PWD/sglang_logs/sglang_cpu"

# ===== WORK_HOME 更稳的写法 =====
WORK_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
echo "WORK_HOME=$WORK_HOME"

# ===== 环境路径 =====
export CONDA_PREFIX="/root/miniforge3/envs/xtang-embedding-cpu"
export SGLANG_USE_CPU_ENGINE=1

# ===== 预装库（安全拼接 LD_PRELOAD）=====
LIBS=(
  "$CONDA_PREFIX/lib/libiomp5.so"
  "$CONDA_PREFIX/lib/libtcmalloc.so"
  "$CONDA_PREFIX/lib/libtbbmalloc.so.2"
)
PRELOAD_JOIN=""
for f in "${LIBS[@]}"; do
  [[ -f "$f" ]] && PRELOAD_JOIN="${PRELOAD_JOIN:+$PRELOAD_JOIN:}$f"
done
export LD_PRELOAD="${PRELOAD_JOIN}${LD_PRELOAD:+:$LD_PRELOAD}"

# ===== 线程/NUMA（按需调整）=====
export MALLOC_ARENA_MAX=1

# ===== BATCH_SIZE 列表 =====
# BATCH_LIST=(1 2 4 8 16 32 64 100 128)
BATCH_LIST=(100)

for BATCH_SIZE in "${BATCH_LIST[@]}"; do
    echo "=============================="
    echo "Running BATCH_SIZE=$BATCH_SIZE"
    echo "=============================="

    numactl -C 0-7,256-263 \
    python $WORK_HOME/main.py \
      --backend sglang-offline \
      --model "$MODEL_DIR" \
      --device cpu \
      --yahoo-jsonl $WORK_HOME/datasets/yahoo_answers_title_answer.jsonl \
      --yahoo-mode q \
      --yahoo-max 10000 \
      --batch-size $BATCH_SIZE \
      --dump-emb $WORK_HOME/runs/yahoo_q_bs${BATCH_SIZE}.pt \
      --output-csv $WORK_HOME/runs/yahoo_eval_bs${BATCH_SIZE}.csv     

done
echo "All done."

# python $WORK_HOME/src/backends/sglang_offline_backend.py --model-path "$MODEL_DIR" --device cpu