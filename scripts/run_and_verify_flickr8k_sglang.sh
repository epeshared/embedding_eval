#!/usr/bin/env bash
set -euo pipefail

# One-shot runner for Flickr8k embeddings + retrieval verification.
#
# Runs:
#   - main.py to generate text/image embeddings (dumped to .pt)
#   - verify_flickr8k_embeddings.py to compute sanity stats + Recall@K
#
# Usage:
#   ./scripts/run_and_verify_flickr8k_sglang.sh offline
#   ./scripts/run_and_verify_flickr8k_sglang.sh online
#
# Common env overrides:
#   BATCH_SIZE=64
#   MAX_SAMPLES=200
#   CAPTIONS_PER_IMAGE=1
#   FLICKR8K_MODALITY=both           # both|text|image
#   FLICKR8K_IMAGES_DIR=/path/to/Flicker8k_Dataset
#   FLICKR8K_CAPTIONS_FILE=/path/to/Flickr8k.token.txt
#   RUNS_DIR=/abs/path/to/runs
#   K_LIST="1,5,10"                  # verification Recall@K
#
# Offline-only env:
#   MODEL_DIR=/abs/path/to/model_dir
#   DEVICE=cpu                       # cpu|cuda
#
# Online-only env:
#   MODEL_PATH=/abs/path/to/served/model   # should match the server's --model-path
#   SGL_URL=http://127.0.0.1:30000
#   SGL_API=v1
#   SGL_API_KEY=
#   SGL_IMAGE_TRANSPORT=data-url          # data-url|base64|path/url
#
# Python override:
#   PYTHON=python

MODE="${1:-${MODE:-offline}}"

WORK_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PYTHON_BIN="${PYTHON:-python}"

FLICKR8K_IMAGES_DIR="${FLICKR8K_IMAGES_DIR:-$WORK_HOME/src/customer/t/datasets/Flickr8k/Flicker8k_Dataset}"
FLICKR8K_CAPTIONS_FILE="${FLICKR8K_CAPTIONS_FILE:-$WORK_HOME/src/customer/t/datasets/Flickr8k/Flickr8k.token.txt}"

BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
CAPTIONS_PER_IMAGE="${CAPTIONS_PER_IMAGE:-1}"
FLICKR8K_MODALITY="${FLICKR8K_MODALITY:-both}"

RUNS_DIR="${RUNS_DIR:-$WORK_HOME/runs}"
mkdir -p "$RUNS_DIR"

K_LIST="${K_LIST:-1,5,10}"

case "$MODE" in
  offline|sglang-offline)
    MODEL_DIR="${MODEL_DIR:-/home/xtang/models/openai/clip-vit-base-patch32/}"
    DEVICE="${DEVICE:-cpu}"
    BACKEND_ARGS=(
      --backend clip
      --model "$MODEL_DIR"
      --device "$DEVICE"
    )
    MODE_TAG="offline_${DEVICE}"
    ;;

  online|sglang-online)
    # IMPORTANT: for SGLang OpenAI-compatible endpoints, the `model` field should
    # match a model name the server recognizes. For local servers started with
    # `--model-path /path/to/model`, using that exact path is the safest default.
    MODEL_PATH="${MODEL_PATH:-/home/xtang/models/openai/clip-vit-base-patch32}"
    SGL_URL="${SGL_URL:-http://127.0.0.1:30000}"
    SGL_API="${SGL_API:-v1}"
    SGL_API_KEY="${SGL_API_KEY:-}"
    SGL_IMAGE_TRANSPORT="${SGL_IMAGE_TRANSPORT:-data-url}"
    BACKEND_ARGS=(
      --backend sglang-online
      --model "$MODEL_PATH"
      --sgl-url "$SGL_URL"
      --sgl-api "$SGL_API"
      --sgl-api-key "$SGL_API_KEY"
      --sgl-image-transport "$SGL_IMAGE_TRANSPORT"
    )
    MODE_TAG="online_${SGL_API}"
    ;;

  *)
    echo "Unknown MODE: '$MODE' (use 'online' or 'offline')" >&2
    exit 2
    ;;
 esac

OUT_PREFIX="$RUNS_DIR/flickr8k_${MODE_TAG}_bs${BATCH_SIZE}_n${MAX_SAMPLES}_cpi${CAPTIONS_PER_IMAGE}"
IMG_PT="${OUT_PREFIX}_img.pt"
TXT_PT="${OUT_PREFIX}_txt.pt"

echo "[Run] mode=$MODE_TAG batch=$BATCH_SIZE max_samples=$MAX_SAMPLES captions_per_image=$CAPTIONS_PER_IMAGE"
echo "[Run] images_dir=$FLICKR8K_IMAGES_DIR"
echo "[Run] captions_file=$FLICKR8K_CAPTIONS_FILE"
echo "[Run] dump_img=$IMG_PT"
echo "[Run] dump_txt=$TXT_PT"

set -x
"$PYTHON_BIN" "$WORK_HOME/main.py" \
  "${BACKEND_ARGS[@]}" \
  --datasets FLICKR8K \
  --flickr8k-images-dir "$FLICKR8K_IMAGES_DIR" \
  --flickr8k-captions-file "$FLICKR8K_CAPTIONS_FILE" \
  --flickr8k-captions-per-image "$CAPTIONS_PER_IMAGE" \
  --flickr8k-modality "$FLICKR8K_MODALITY" \
  --max-samples "$MAX_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --dump-img-emb "$IMG_PT" \
  --dump-txt-emb "$TXT_PT"
set +x

echo "[Verify] Recall@K=$K_LIST"
"$WORK_HOME/scripts/verify_flickr8k_embeddings.sh" \
  --img-pt "$IMG_PT" \
  --txt-pt "$TXT_PT" \
  --captions-per-image "$CAPTIONS_PER_IMAGE" \
  --k "$K_LIST" \
  --max-n "$MAX_SAMPLES"
