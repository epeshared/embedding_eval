# 强烈建议在 GPU 环境里卸掉 IPEX，避免噪声
# pip uninstall -y intel-extension-for-pytorch

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
# export VLLM_USE_FA2=0

WORK_HOME=$PWD/../
echo "WORK_HOME=$WORK_HOME"
BATCH_SIZE=400

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python - <<'PY'
import torch
torch.cuda.empty_cache()
PY

export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY

MODEL="Qwen/Qwen3-Embedding-4B"
echo "Using model: $MODEL_DIR"

numactl -C 0-7 \
python $WORK_HOME/main.py \
  --backend vllm_openai \
  --model $MODEL \
  --batch-size $BATCH_SIZE \
  --yahoo-jsonl $WORK_HOME/datasets/yahoo_answers_title_answer.jsonl \
  --yahoo-mode q \
  --yahoo-max 10000 \
  --dump-emb runs/yahoo_q_10k.pt \
  --output-csv runs/yahoo_eval.csv \
  --vllm_openai-url "$OPENAI_BASE_URL"


