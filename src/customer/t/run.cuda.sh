#!/usr/bin/env bash
set -euo pipefail

# GPU
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# 要测试的 batch_size 列表（随便改）
BATCH_LIST=(1 2 4 8 16 32 64 100 128)

DATA_TYPE="fp16"
NUM_ITER=100
PARALLELISM=2
DEVICE="auto"
MODEL="/home/xtang/modles/openai/clip-vit-base-patch32"

echo "===== CLIP Benchmark Loop Start ====="
echo "Model: $MODEL"
echo "DATA_TYPE=$DATA_TYPE, NUM_ITER=$NUM_ITER, PARALLELISM=$PARALLELISM, DEVICE=$DEVICE"
echo ""

for BATCH_SIZE in "${BATCH_LIST[@]}"; do
    echo "------------------------------------------------------------"
    echo "Running batch_size = $BATCH_SIZE"
    echo "------------------------------------------------------------"

    python benchmark_clip.orig.py \
        --data_type=$DATA_TYPE \
        --num_iter=$NUM_ITER \
        --parallelism=$PARALLELISM \
        --batch_size=$BATCH_SIZE \
        --device=$DEVICE \
        --model $MODEL

    echo ""
done

echo "===== Benchmark Finished ====="


# Flickr8k
# python benchmark_clip.py \
#   --data_source flickr8k \
#   --flickr_images_dir "./datasets/Flickr8k/Flicker8k_Dataset/" \
#   --flickr_captions_file "./datasets/Flickr8k/Flickr8k.token.txt" \
#   --parallelism 2 --batch_size 100 --num_iter 100 --data_type=fp16 \
#   --max_samples 20000 \
# #  --copy_per_iter
