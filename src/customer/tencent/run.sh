#!/usr/bin/env bash
set -euo pipefail

# random
numactl -C 0-15 python benchmark_clip.py --data_type=bf16 --num_iter=100 --parallelism=2 --batch_size=100 --device=cpu 
# numactl -C 0-15 python benchmark_clip.py --data_type=fp16 --num_iter=100 --parallelism=2 --batch_size=100 --device=cpu --copy_per_iter
#

# Flickr8k
# numactl -C 0-15 python benchmark_clip.py \
#   --data_source flickr8k \
#   --flickr_images_dir "./datasets/Flickr8k/Flicker8k_Dataset/" \
#   --flickr_captions_file "./datasets/Flickr8k/Flickr8k.token.txt" \
#   --parallelism 2 --batch_size 100 --num_iter 100 --data_type=fp16 \
#   --device=cpu \
#   --max_samples 20000 \
#   --copy_per_iter

