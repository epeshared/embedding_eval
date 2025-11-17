#!/usr/bin/env bash
set -euo pipefail

Flickr8k_HOME="/home/xtang/vdb-sandbox/models/embedding/src/customer/tencent/datasets/Flickr8k/"


# python benchmark_clip_sglang.py \
#   --mode multimodal \
#   --data_source flickr8k \
#   --base_url=http://127.0.0.1:30000 \
#   --model=openai/clip-vit-large-patch14-336 \
#   --api=v1 \
#   --flickr_images_dir "$Flickr8k_HOME/Flicker8k_Dataset" \
#   --flickr_captions_file "$Flickr8k_HOME/Flickr8k.token.txt" \
#   --validate \
#   --validate_samples 1 \
#   --validate_group_size 5 \
#   --validation_dump ./sglang_mm_validation.txt \
#   --validation_distractors 2 \
#   --validate_start_group 58 \
#   --warmup --warmup_iters 1

python benchmark_clip_sglang.py \
  --mode multimodal \
  --data_source flickr8k \
  --base_url=http://127.0.0.1:30000 \
  --model=openai/clip-vit-base-patch32 \
  --api=v1 \
  --flickr_images_dir "$Flickr8k_HOME/Flicker8k_Dataset" \
  --flickr_captions_file "$Flickr8k_HOME/Flickr8k.token.txt" \
  --validate \
  --validate_samples 1 \
  --validate_group_size 5 \
  --validation_dump ./sglang_mm_validation.txt \
  --validation_distractors 2 \
  --validate_start_group 58 \
  --warmup --warmup_iters 1