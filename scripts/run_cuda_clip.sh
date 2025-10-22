numactl -C 0 python ../main.py  \
  --backend clip \
  --model openai/clip-vit-base-patch32 \
  --device cuda --amp auto \
  --datasets food101 --batch-size 16
