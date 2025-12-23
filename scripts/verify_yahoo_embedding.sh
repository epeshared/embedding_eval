# 按 text 对齐，算 cosine：

# python cverify_yahoo_embedding.py \
#   --a outputs/yahoo_emb_a.jsonl \
#   --b outputs/yahoo_emb_b.jsonl \
#   --key text \
#   --metric cosine \
#   --threshold 0.999 \
#   --topk 20 \
#   --out outputs/compare_cosine.csv


# 按 idx 对齐（有时候 text 可能重复/不完全一致）：

# python verify_yahoo_embedding.py \
#   --a outputs/yahoo_emb_a.jsonl \
#   --b outputs/yahoo_emb_b.jsonl \
#   --key idx \
#   --metric cosine \
#   --threshold 0.999 \
#   --topk 20


# 如果你想看数值“差异有多大”，用 L2：

python /home/xtang/embedding_eval/scripts/verify_yahoo_embedding.py \
  --a /home/xtang/embedding_eval/outputs/yahoo_emb_bs10_run1.jsonl \
  --b /home/xtang/embedding_eval/outputs/yahoo_emb_bs10_run1.jsonl \
  --key text \
  --metric l2 \
  --threshold 0.1 \
  --topk 20