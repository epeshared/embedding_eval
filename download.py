from huggingface_hub import snapshot_download
import os

# 指向你的镜像（若镜像需要，设置 env）
os.environ["HUGGINGFACE_HUB_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 强制单线程下载，避免 HEAD 并发导致的问题
path = snapshot_download(
    repo_id="Qwen/Qwen3-Embedding-4B",
    cache_dir="models/Qwen/Qwen3-Embedding-4B/.cache/huggingface",
    max_workers=1,           # 关键：不要并行
    force_download=False,    # 如果需要可设 True
    local_files_only=False
)
print("snapshot at:", path)

