# numactl -C 0-7 python ../main.py \
#   --backend clip \
#   --model openai/clip-vit-base-patch32 \
#   --datasets food101 \
#   --offline True \
#   --batch-size 16 \
#   --amx off --amx-verbose False \
#   --device cpu --use-ipex True --amp bf16 \
#   --output-csv ../runs/clip_food101.csv

WORK_HOME=$PWD/../
echo "WORK_HOME=$WORK_HOME"

numactl -C 0-15 python $WORK_HOME/main.py \
  --backend clip \
  --model openai/clip-vit-base-patch32 \
  --datasets food101 \
  --offline True \
  --batch-size 16 \
  --amx on --amx-verbose False \
  --device cpu --use-ipex True --amp bf16 \
  --output-csv ../runs/clip_food101_AMX.csv
