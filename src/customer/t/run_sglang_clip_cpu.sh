#!/usr/bin/env bash
set -euo pipefail

WORK_HOME=$PWD/../
echo "WORK_HOME=$WORK_HOME"
###############################################
#        ✅ 仅需在这里配置模型路径即可
###############################################
MODEL="openai/clip-vit-base-patch32"
clip_variant="base"
# MODEL="openai/clip-vit-large-patch14-336"    
# clip_variant="large-336"
###############################################
echo "Using model: $MODEL"
echo "clip_variant: $clip_variant"

WORKERS=2
CORES="16-17"
echo "Using workers: $WORKERS"
echo "Using cores: $CORES"

BATCH_LIST=(1 2 4 8 16 32 64 100 128)

for BATCH_SIZE in "${BATCH_LIST[@]}"; do
    echo "=============================="
    echo "Running BATCH_SIZE=$BATCH_SIZE"
    echo "=============================="

    python benchmark_clip_sglang_launcher.py \
      --workers $WORKERS \
      --cores $CORES \
      --script benchmark_clip_sglang.py \
      --logs_dir logs\batchsize_${BATCH_SIZE} \
      -- \
      --base_url=http://127.0.0.1:30000 \
      --model=$MODEL \
      --api=v1 \
      --mode=multimodal \
      --data_source=random \
      --image_transport=data-url \
      --num_samples=10000 \
      --batch_size=$BATCH_SIZE \
      --clip_variant $clip_variant

done
echo "All done."
