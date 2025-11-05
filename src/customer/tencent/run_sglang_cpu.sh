#!/usr/bin/env bash
set -euo pipefail

#纯文本 + 随机句子
# python benchmar_clip_sglang.py \
#   --base_url http://127.0.0.1:30000 \
#   --model Qwen/Qwen3-Embedding-4B \
#   --api v1 \
#   --mode text --data_source random \
#   --num_samples 1024 --batch_size 128

#文本 + 图片（多模态），随机图片与固定文本
# python benchmar_clip_sglang.py \
#   --base_url http://127.0.0.1:30000 \
#   --model openai/clip-vit-large-patch14-336 \
#   --api v1 \
#   --mode multimodal --data_source random \
#   --image_transport data-url \
#   --num_samples 10000 --batch_size 100


#Flickr8k，parts 负载（path/url）
# python benchmar_clip_sglang.py \
#   --base_url http://127.0.0.1:30000 \
#   --model gme-qwen2-vl \
#   --api v1 \
#   --mode multimodal --data_source flickr8k \
#   --flickr_images_dir /path/Flicker8k_Dataset \
#   --flickr_captions_file /path/Flickr8k_text/Flickr8k.token.txt \
#   --flickr_caption_pick random \
#   --num_samples 512 --batch_size 64


# python benchmark_clip_sglang_launcher.py \
#   --workers 4 \
#   --cores 16-19 \
#   --script benchmark_clip_sglang.py \
#   -- \
#   --base_url=http://127.0.0.1:30000 \
#   --model=openai/clip-vit-large-patch14-336 \
#   --api=v1 \
#   --mode=multimodal \
#   --data_source=random \
#   --image_transport=data-url \
#   --num_samples=10000 \
#   --batch_size=100

# python benchmark_clip_sglang_launcher.py \
#   --workers 2 \
#   --cores 16-17 \
#   --script benchmark_clip_sglang.py \
#   -- \
#   --base_url=http://127.0.0.1:30000 \
#   --model=openai/clip-vit-base-patch32 \
#   --api=v1 \
#   --mode=multimodal \
#   --data_source=random \
#   --image_transport=data-url \
#   --num_samples=10000 \
#   --batch_size=100

python benchmark_clip_sglang_launcher.py \
  --workers 2 \
  --cores 16-17 \
  --script benchmark_clip_sglang.py \
  -- \
  --base_url=http://127.0.0.1:30000 \
  --model=openai/clip-vit-large-patch14-336 \
  --api=v1 \
  --mode=multimodal \
  --data_source=random \
  --image_transport=data-url \
  --num_samples=10000 \
  --batch_size=100 \
  --clip_variant large-336
