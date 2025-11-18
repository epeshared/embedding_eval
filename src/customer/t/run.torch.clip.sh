#!/usr/bin/env bash
set -euo pipefail

# GPU
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# ===== 可配置参数 =====
TOTAL_IMAGES=20000
PARALLELISM=2
DATA_TYPE=fp16
DEVICE=auto
MODEL="openai/clip-vit-base-patch32"

# 要 sweep 的 batch size 列表（随便加）
BATCH_SIZE_LIST=(1 2 4 8 16 32 64 100 128)

echo "========== Batch Size Sweep =========="
echo "TOTAL_IMAGES=${TOTAL_IMAGES}"
echo "PARALLELISM=${PARALLELISM}"
echo "DATA_TYPE=${DATA_TYPE}"
echo "DEVICE=${DEVICE}"
echo ""

for BATCH_SIZE in "${BATCH_SIZE_LIST[@]}"; do
    per_step=$((PARALLELISM * BATCH_SIZE))

    # total_images 必须能整除 per_step
    if (( TOTAL_IMAGES % per_step != 0 )); then
        echo "error! batch_size=${BATCH_SIZE}, 因为 TOTAL_IMAGES % (parallelism*batch_size) != 0"
        exit 1
    fi

    echo "------------------------------------------------------------"
    echo "Running batch_size = ${BATCH_SIZE}"
    echo "------------------------------------------------------------"

    python bench_clip.py \
      --data_type=${DATA_TYPE} \
      --parallelism=${PARALLELISM} \
      --batch_size=${BATCH_SIZE} \
      --total_images=${TOTAL_IMAGES} \
      --device=${DEVICE} \
      --model=${MODEL}

    echo ""
done

echo "========== Sweep Finished =========="
