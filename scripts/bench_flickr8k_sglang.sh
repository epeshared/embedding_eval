#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./bench_flickr8k_sglang.sh online
#   ./bench_flickr8k_sglang.sh offline
#
# Common env overrides:
#   BATCH_LIST_STR="1 2 4 8 16 32 64 128"
#   MAX_SAMPLES=1000                 # used for Flickr8k: max number of images
#   CAPTIONS_PER_IMAGE=1             # Flickr8k typically has 5
#   FLICKR8K_MODALITY=both            # both|text|image (default both)
#   WARMUP=-1                        # warmup samples; <=0 uses default
#   PROFILE=0                        # set 1/true to enable profiling
#   FLICKR8K_IMAGES_DIR=/path/to/Flicker8k_Dataset
#   FLICKR8K_CAPTIONS_FILE=/path/to/Flickr8k.token.txt
#
# Online-only env:
#   MODEL_PATH="Qwen/Qwen3-Embedding-4B"
#   SGL_URL="http://127.0.0.1:30000"
#   SGL_API="v1"                    # v1|native|openai (image embedding requires v1/openai)
#   SGL_API_KEY=""
#
# Offline-only env:
#   MODEL_DIR="/abs/path/to/model_dir"
#   DEVICE="cpu"                    # cpu|cuda

MODE="${1:-${MODE:-offline}}"

WORK_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
echo "WORK_HOME=$WORK_HOME"

FLICKR8K_IMAGES_DIR="${FLICKR8K_IMAGES_DIR:-$WORK_HOME/src/customer/t/datasets/Flickr8k/Flicker8k_Dataset}"
FLICKR8K_CAPTIONS_FILE="${FLICKR8K_CAPTIONS_FILE:-$WORK_HOME/src/customer/t/datasets/Flickr8k/Flickr8k.token.txt}"

BATCH_LIST_STR="${BATCH_LIST_STR:-100}"
read -r -a BATCH_LIST <<<"$BATCH_LIST_STR"

MAX_SAMPLES="${MAX_SAMPLES:-1000}"
CAPTIONS_PER_IMAGE="${CAPTIONS_PER_IMAGE:-1}"
FLICKR8K_MODALITY="${FLICKR8K_MODALITY:-both}"
WARMUP="${WARMUP:--1}"
PROFILE="${PROFILE:-0}"

RUNS_DIR="${RUNS_DIR:-$WORK_HOME/runs}"
LOG_DIR="${LOG_DIR:-$WORK_HOME/scripts/logs}"
mkdir -p "$RUNS_DIR" "$LOG_DIR"

BASE_ARGS=(
  --datasets FLICKR8K
  --flickr8k-images-dir "$FLICKR8K_IMAGES_DIR"
  --flickr8k-captions-file "$FLICKR8K_CAPTIONS_FILE"
  --flickr8k-captions-per-image "$CAPTIONS_PER_IMAGE"
  --flickr8k-modality "$FLICKR8K_MODALITY"
  --warmup "$WARMUP"
  --max-samples "$MAX_SAMPLES"
)

if [[ "${PROFILE,,}" == "1" || "${PROFILE,,}" == "true" ]]; then
  BASE_ARGS+=(--profile)
fi

case "$MODE" in
  online|sglang-online)
    MODEL_PATH="${MODEL_PATH:-openai/clip-vit-base-patch32}"
    SGL_URL="${SGL_URL:-http://127.0.0.1:30000}"
    SGL_API="${SGL_API:-v1}"
    SGL_API_KEY="${SGL_API_KEY:-}"

    BACKEND_ARGS=(
      --backend sglang-online
      --model "$MODEL_PATH"
      --sgl-url "$SGL_URL"
      --sgl-api "$SGL_API"
      --sgl-api-key "$SGL_API_KEY"
    )
    MODE_TAG="online_${SGL_API}"
    ;;

  offline|sglang-offline)
    MODEL_DIR="${MODEL_DIR:-/home/xtang/models/openai/clip-vit-base-patch32/}"
    DEVICE="${DEVICE:-cpu}"

    echo "[Probe] checking if sglang-offline is usable..."
    if python - <<'PY'
import importlib
import sys

try:
    # sglang-offline needs the sglang runtime (srt/Engine) plus multimodal processors.
    import sglang.srt  # noqa: F401
    import vllm  # noqa: F401

    # Ensure the CLIP multimodal processor module imports (this will fail if optional deps are missing
    # or the platform cannot load required kernels).
    importlib.import_module("sglang.srt.multimodal.processors.clip")

    # Ensure the processor registry contains CLIPModel.
    from sglang.srt.managers import multimodal_processor as mp

    mp.import_processors("sglang.srt.multimodal.processors")
    has_clip = any(getattr(k, "__name__", "") == "CLIPModel" for k in mp.PROCESSOR_MAPPING.keys())
    if not has_clip:
        raise RuntimeError(
            "CLIPModel multimodal processor not registered; sglang-offline can't encode images"
        )
except Exception as e:
    print(f"SGLANG_OFFLINE_UNAVAILABLE: {e}", file=sys.stderr)
    sys.exit(1)

sys.exit(0)
PY
    then
      BACKEND_ARGS=(
        # Offline (Engine) mode.
        # This uses sglang.Engine locally (no HTTP server), and supports both text + image embeddings.
        --backend sglang-offline
        --model "$MODEL_DIR"
        --device "$DEVICE"
      )
      MODE_TAG="offline_${DEVICE}_sglang"
    else
      echo "[Warn] sglang-offline deps not available; falling back to local clip backend." >&2
      BACKEND_ARGS=(
        --backend clip
        --model "$MODEL_DIR"
        --device "$DEVICE"
      )
      MODE_TAG="offline_${DEVICE}_clip"
    fi
    ;;

  *)
    echo "Unknown MODE: '$MODE' (use 'online' or 'offline')" >&2
    exit 1
    ;;
esac

for BATCH_SIZE in "${BATCH_LIST[@]}"; do
  echo "=============================="
  echo "Flickr8k bench: mode=$MODE_TAG batch_size=$BATCH_SIZE max_samples=$MAX_SAMPLES captions_per_image=$CAPTIONS_PER_IMAGE"
  echo "images_dir=$FLICKR8K_IMAGES_DIR"
  echo "captions_file=$FLICKR8K_CAPTIONS_FILE"
  echo "=============================="

  OUT_CSV="$RUNS_DIR/flickr8k_${MODE_TAG}_bs${BATCH_SIZE}.csv"
  OUT_JSONL="$RUNS_DIR/flickr8k_${MODE_TAG}_bs${BATCH_SIZE}.jsonl"
  LOG_FILE="$LOG_DIR/flickr8k_${MODE_TAG}_bs${BATCH_SIZE}.log"

  echo "[Run] logging to: $LOG_FILE"
  # Stream output to screen and also save to log.
  python "$WORK_HOME/main.py" \
    "${BACKEND_ARGS[@]}" \
    --batch-size "$BATCH_SIZE" \
    --output-csv "$OUT_CSV" \
    --output-jsonl "$OUT_JSONL" \
    --use-ipex true\
    "${BASE_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"

  echo "Saved: $OUT_CSV"
  echo "Log:   $LOG_FILE"
done

echo "All done."