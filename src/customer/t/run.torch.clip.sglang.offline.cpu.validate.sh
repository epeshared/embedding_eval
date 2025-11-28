#!/usr/bin/env bash
set -euo pipefail

# CPU
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
export CONDA_PREFIX="/home/yanbingj/miniforge3/envs/embed_eval"
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

# ===== 可配置参数 =====
TOTAL_IMAGES=1
PARALLELISM=1
DATA_TYPE=fp16
DEVICE=cpu
# MODEL="openai/clip-vit-base-patch32"
MODEL="openai/clip-vit-large-patch14-336"

# 要 sweep 的 batch size 列表（随便加）
#BATCH_SIZE_LIST=(1 2 4 8 16 32 64 100 128)
BATCH_SIZE_LIST=(100)

# numactl -C 0-7 \
python bench_clip_sglang_offline.py \
      --validate
exit 0
