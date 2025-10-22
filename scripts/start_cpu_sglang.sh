#!/usr/bin/env bash
set -euo pipefail

# ===== OneDNN / IPEX 建议 =====
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
export DNNL_VERBOSE=0
export IPEX_DISABLE_AUTOCAST=1   # 建议开启，规避 uint64 copy_kernel 坑

# ===== 日志目录 =====
mkdir -p "sglang_logs/sglang_cpu"
export SGLANG_TORCH_PROFILER_DIR="$PWD/sglang_logs/sglang_cpu"

# ===== WORK_HOME 更稳的写法 =====
WORK_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
echo "WORK_HOME=$WORK_HOME"

# ===== 环境路径 =====
export CONDA_PREFIX="/root/miniforge3/envs/xtang-embedding-cpu"
export SGLANG_USE_CPU_ENGINE=1

# ===== 预装库（安全拼接 LD_PRELOAD）=====
# 说明：tcmalloc 与 tbbmalloc 不建议同时启用，二选一即可。保留你原先两者都上的写法，但若遇到冲突，可只留其一。
LIBS=(
  "$CONDA_PREFIX/lib/libiomp5.so"
  "$CONDA_PREFIX/lib/libtcmalloc.so"
  "$CONDA_PREFIX/lib/libtbbmalloc.so.2"
)
# 过滤掉不存在的库，避免意外
PRELOAD_JOIN=""
for f in "${LIBS[@]}"; do
  [[ -f "$f" ]] && PRELOAD_JOIN="${PRELOAD_JOIN:+$PRELOAD_JOIN:}$f"
done
# 条件追加已有 LD_PRELOAD
export LD_PRELOAD="${PRELOAD_JOIN}${LD_PRELOAD:+:$LD_PRELOAD}"

# ===== 线程/NUMA（按需调整）=====
export OMP_NUM_THREADS=16
export MALLOC_ARENA_MAX=1

# ===== Batch Size =====
BATCH_SIZE=16
echo "Batch size = $BATCH_SIZE"

# ===== 绑核与启动 =====
numactl -C 0-15 \
python -m sglang.launch_server \
  --model-path "$WORK_HOME/models/Qwen/Qwen3-Embedding-4B" \
  --tokenizer-path "$WORK_HOME/models/Qwen/Qwen3-Embedding-4B" \
  --trust-remote-code \
  --disable-overlap-schedule \
  --is-embedding \
  --device cpu \
  --host 0.0.0.0 --port 30000 \
  --skip-server-warmup \
  --tp 1 \
  --enable-torch-compile \
  --torch-compile-max-bs "$BATCH_SIZE" \
  --attention-backend intel_amx \
  --log-level error
