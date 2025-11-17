#!/usr/bin/env bash
set -euo pipefail

# ========= workspace =========
WORK_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
echo "WORK_HOME=$WORK_HOME"

###############################################
#        ✅ 仅需在这里配置模型路径即可
###############################################
MODEL_NAME="Qwen/Qwen3-Embedding-4B"
MODEL_DIR="$WORK_HOME/models/${MODEL_NAME}"
# MODEL_DIR="$WORK_HOME/models/Qwen/Qwen3-Embedding-0.6B"
# MODEL_DIR="$WORK_HOME/models/BAAI/bge-large-zh-v1.5"
###############################################
echo "Using model: $MODEL_DIR"

# ========= 端口 / API Key =========
HOST="0.0.0.0"
PORT="8000"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"   # 可外部 export 覆盖
UVICORN_LOG_LEVEL="error"               # 新版用这个替代 --log-level

# ========= dtype / 并行 / 绑核 =========
DTYPE="bfloat16"          # 如 CPU 不支持 bf16，请改为 fp32
TP=1
MAX_MODEL_LEN=8192
CPU_CORES="0-15"

# ========= 日志目录 =========
mkdir -p "$WORK_HOME/vllm_logs/cuda"
echo "Logs -> $WORK_HOME/vllm_logs/cuda"

# ========= 依赖与版本探测 =========
read -r VLLM_IMPORT_OK VLLM_VER <<<"$(python - <<'PY'
import sys
try:
    import vllm, re
    ver = getattr(vllm, "__version__", "0.0.0")
    print("OK", ver)
except Exception as e:
    print("NO", "0.0.0")
PY
)"

if [[ "$VLLM_IMPORT_OK" != "OK" ]]; then
  echo "[FATAL] vLLM not installed or import failed."
  echo "Fix: pip install vllm --extra-index-url https://download.pytorch.org/whl/cu126  # 或 cpu"
  exit 1
fi

echo "Detected vLLM version: $VLLM_VER"

# 比较是否为 0.6.0 及以上
is_ge_060="$(python - <<PY
from packaging.version import Version
import os
ver = os.environ.get("VLLM_VER","0.0.0")
print( Version(ver) >= Version("0.6.0") )
PY
)"
export VLLM_VER

# ========= 组装通用参数（新旧版本通用）=========
COMMON_ARGS=(
  --model "$MODEL_DIR"
  --tensor-parallel-size "$TP"
  --host "$HOST"
  --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
  --dtype "$DTYPE"
  --served-model-name "$MODEL_NAME"
  --max-num-seqs 500 \        # v0 默认就是 256，可按显存提升
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.90
)

# ========= 新旧版本差异参数 =========
VLLM_MODE_ARGS=()
if [[ "$is_ge_060" == "True" ]]; then
  # vLLM >= 0.6：不再使用 --task；采用 --runner pooling 和 --convert embed
  VLLM_MODE_ARGS+=( --runner pooling --convert embed )
  # 日志级别参数名字也变了
  UVICORN_ARGS=( --uvicorn-log-level "$UVICORN_LOG_LEVEL" )
  echo "[Info] Using new CLI: --runner pooling --convert embed"
else
  # vLLM < 0.6：保留旧的 --task embed（已 deprecated，但可用）
  VLLM_MODE_ARGS+=( --task embed )
  UVICORN_ARGS=()  # 老版本没有 --uvicorn-log-level；会忽略旧的 --log-level
  echo "[Info] Using legacy CLI: --task embed"
fi

# ========= 设备选择说明 =========
# 新版 CLI 不再支持 --device 参数；设备由 vLLM 自动选择：
# - 有 CUDA 时默认用 GPU
# - 强制 CPU：运行前 export CUDA_VISIBLE_DEVICES=""
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "Note: device is auto-selected by vLLM. To force CPU: export CUDA_VISIBLE_DEVICES=\"\""

# ========= torch.compile 模式启用 =========
export VLLM_TORCH_COMPILE=1
export VLLM_TORCH_COMPILE_MODE=max-autotune  # or reduce-overhead

# 或者使用 CUDA Graph
export VLLM_USE_CUDA_GRAPH=1

# ========= 执行（GPU 上 numactl 无意义；保留以兼容 CPU 环境）=========
echo "Starting vLLM OpenAI server (Embeddings) on ${HOST}:${PORT} ..."
set -x
numactl -C "${CPU_CORES}" \
python -m vllm.entrypoints.openai.api_server \
  "${COMMON_ARGS[@]}" \
  "${VLLM_MODE_ARGS[@]}" \
  "${UVICORN_ARGS[@]}"
set +x
