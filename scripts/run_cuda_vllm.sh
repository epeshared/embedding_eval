# 强烈建议在 GPU 环境里卸掉 IPEX，避免噪声
# pip uninstall -y intel-extension-for-pytorch

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
# export VLLM_USE_FA2=0

WORK_HOME=$PWD/../
echo "WORK_HOME=$WORK_HOME"
BATCH_SIZE=100

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python - <<'PY'
import torch
torch.cuda.empty_cache()
PY

# numactl -C 0-8 python embedding_bench.py \
#  --backend vllm \
#  --model Qwen/Qwen3-Embedding-4B \
#  --vllm-dtype half \
#  --vllm-tp 1 \
#  --offline True \
#  --vllm-device cuda \
#  --vllm-max-model-len 8192 \
#  --vllm-gpu-mem-util 0.92 \
#  --datasets LCQMC \
#  --batch-size 16 \
#  --use-ipex False

numactl -C 0-7 python $WORK_HOME/main.py \
  --backend vllm \
  --model Qwen/Qwen3-Embedding-4B \
  --vllm-device cuda \
  --vllm-dtype float16 \
  --batch-size $BATCH_SIZE \
  --yahoo-jsonl $WORK_HOME/datasets/yahoo_answers_title_answer.jsonl \
  --yahoo-mode q \
  --yahoo-max 10000 \
  --dump-emb runs/yahoo_q_10k.pt \
  --output-csv runs/yahoo_eval.csv

