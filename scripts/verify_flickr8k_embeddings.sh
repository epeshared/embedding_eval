#!/usr/bin/env bash
set -euo pipefail

# Verify Flickr8k text/image embeddings produced by main.py --dump-img-emb/--dump-txt-emb.
#
# Usage:
#   ./scripts/verify_flickr8k_embeddings.sh \
#     --img-pt runs/f8k_img.pt \
#     --txt-pt runs/f8k_txt.pt \
#     --captions-per-image 1
#
# Optional:
#   --k "1,5,10"        # Recall@K list
#   --max-n 200          # only evaluate on first N images
#
# Env overrides:
#   PYTHON=python

WORK_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PYTHON_BIN="${PYTHON:-python}"

IMG_PT=""
TXT_PT=""
CAP_PER_IMG=""
K_LIST="1,5,10"
MAX_N="-1"

usage() {
  cat <<EOF
Usage:
  $0 --img-pt <path> --txt-pt <path> --captions-per-image <int> [--k "1,5,10"] [--max-n N]

Examples:
  $0 --img-pt runs/f8k_img.pt --txt-pt runs/f8k_txt.pt --captions-per-image 1
  $0 --img-pt runs/f8k_img.pt --txt-pt runs/f8k_txt.pt --captions-per-image 5 --max-n 200
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --img-pt)
      IMG_PT="$2"; shift 2 ;;
    --txt-pt)
      TXT_PT="$2"; shift 2 ;;
    --captions-per-image)
      CAP_PER_IMG="$2"; shift 2 ;;
    --k)
      K_LIST="$2"; shift 2 ;;
    --max-n)
      MAX_N="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$IMG_PT" || -z "$TXT_PT" || -z "$CAP_PER_IMG" ]]; then
  echo "Missing required args." >&2
  usage
  exit 2
fi

# Allow relative paths from repo root.
if [[ "$IMG_PT" != /* ]]; then IMG_PT="$WORK_HOME/$IMG_PT"; fi
if [[ "$TXT_PT" != /* ]]; then TXT_PT="$WORK_HOME/$TXT_PT"; fi

if [[ ! -f "$IMG_PT" ]]; then
  echo "Image embedding file not found: $IMG_PT" >&2
  exit 1
fi
if [[ ! -f "$TXT_PT" ]]; then
  echo "Text embedding file not found: $TXT_PT" >&2
  exit 1
fi

set -x
"$PYTHON_BIN" "$WORK_HOME/scripts/verify_flickr8k_embeddings.py" \
  --img-pt "$IMG_PT" \
  --txt-pt "$TXT_PT" \
  --captions-per-image "$CAP_PER_IMG" \
  --k "$K_LIST" \
  --max-n "$MAX_N"
