# 强烈建议在 GPU 环境里卸掉 IPEX，避免噪声
# pip uninstall -y intel-extension-for-pytorch

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
# export VLLM_USE_FA2=0

WORK_HOME=$PWD/../
echo "WORK_HOME=$WORK_HOME"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python - <<'PY'
import torch
torch.cuda.empty_cache()
PY


MODEL="Qwen/Qwen3-Embedding-4B"
echo "Using model: $MODEL"

# BATCH_LIST=(1 2 4 8 16 32 64 100 128)
BATCH_LIST=(100)

for BATCH_SIZE in "${BATCH_LIST[@]}"; do
    echo "=============================="
    echo "Running BATCH_SIZE=$BATCH_SIZE"
    echo "=============================="

    # numactl -C 0-8 
    python $WORK_HOME/main.py \
      --backend vllm \
      --model MODEL \
      --vllm-dtype bfloat16 \
      --vllm-device cuda \
      --vllm-tp 1 \
      --vllm-gpu-mem-util 0.90 \
      --yahoo-jsonl $WORK_HOME/datasets/yahoo_answers_title_answer.jsonl \
      --yahoo-mode q \
      --yahoo-max 10000 \
      --batch-size $BATCH_SIZE

done
echo "All done."


