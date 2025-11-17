#!/usr/bin/env bash
set -euo pipefail

# GPU: 
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# random
#numactl -C 0-7 python benchmark_clip.py --data_type=fp16 --num_iter=100 --parallelism=2 --batch_size=100 --device=auto 
#numactl -C 0-7 python benchmark_clip.py --data_type=fp16 --num_iter=100 --parallelism=2 --batch_size=100 --device=auto --copy_per_iter
#

# Flickr8k
# python benchmark_clip.py \
#   --data_source flickr8k \
#   --flickr_images_dir "./datasets/Flickr8k/Flicker8k_Dataset/" \
#   --flickr_captions_file "./datasets/Flickr8k/Flickr8k.token.txt" \
#   --parallelism 2 --batch_size 100 --num_iter 100 --data_type=fp16 \
#   --max_samples 20000 \
# #  --copy_per_iter

