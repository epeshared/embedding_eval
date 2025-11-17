# CLIP Benchmark & Benchmarking Guide

This README focuses on running and benchmarking CLIP (text & image embeddings) within this repository. Zero‑shot dataset specifics (e.g. Food‑101) are intentionally omitted to keep the guide backend‑agnostic.

---
## Contents
- Overview
- Installation
- Benchmark Tools (SGLang client scripts)
- Multi‑process & CPU core pinning
- Flickr8k Validation Mode
- Random Synthetic Benchmark Mode
- Output & Metrics
- Performance Tuning (CPU/GPU)
- Troubleshooting
- Extending & Exporting Embeddings

---
## 1. Overview
We provide two primary ways to work with CLIP:
1. Local embedding extraction using `main.py`.
2. Remote SGLang server benchmarking (text, multimodal image+text) using helper scripts under `src/customer/tencent/`.

Models supported include `openai/clip-vit-base-patch32` and `openai/clip-vit-large-patch14-336` (optionally via IPEX on CPU for acceleration).

---
## 2. Installation
Minimal packages (if not already installed):
```bash
pip install torch transformers pillow requests
```
Optional acceleration / features:
```bash
pip install intel-extension-for-pytorch   # IPEX (CPU acceleration)
```
If you intend to use SGLang server mode (remote embeddings):
```bash
pip install openai  # Only if using --sgl-api openai
```

---
## 3. Benchmark Tools (Remote SGLang)
Located under `src/customer/tencent/`:
- `benchmark_clip_sglang.py`: Sends embedding or multimodal (image+text) requests to a running SGLang server.
- `benchmark_clip_sglang_launcher.py`: Launches multiple worker processes pinned to specific CPU cores.
- `run_sglang_clip_cpu.sh`: Iterates through `BATCH_SIZE` list for random synthetic multimodal workload.
- `run_sglang_clip_val.sh`: Flickr8k validation style benchmark.

### Basic remote run
```bash
python benchmark_clip_sglang.py \
  --mode multimodal \
  --data_source random \
  --base_url http://127.0.0.1:30000 \
  --model openai/clip-vit-base-patch32 \
  --api v1 \
  --num_samples 1000 \
  --batch_size 64 \
  --image_transport data-url
```
Parameters:
- `--api`: `v1 | native | openai` depending on server endpoint style.
- `--mode`: `text` or `multimodal`.
- `--data_source`: `random` (synthetic) or `flickr8k` (real images/captions).
- `--clip_variant`: choose resolution preset for synthetic images: `base` or `large-336`.
- `--image_transport`: `data-url` (base64 data URI) vs `path/url` (send paths or URLs if server can open them).

---
## 4. Multi‑process & CPU Core Pinning
Use `benchmark_clip_sglang_launcher.py` to spawn multiple processes, each bound to a specific CPU core to avoid contention.

Example (2 workers on cores 16-17):
```bash
python benchmark_clip_sglang_launcher.py \
  --workers 2 --cores 16-17 \
  --script benchmark_clip_sglang.py \
  --logs_dir logs_pinned \
  -- --base_url http://127.0.0.1:30000 --model openai/clip-vit-base-patch32 \
     --api v1 --mode multimodal --data_source random --num_samples 10000 --batch_size 100
```
Dry run (no processes launched):
```bash
python benchmark_clip_sglang_launcher.py --workers 4 --dry-run -- --base_url http://127.0.0.1:30000 --model openai/clip-vit-base-patch32
```
If your environment lacks `sched_setaffinity`, use `--use-taskset` for pinning.

Internal environment defaults set each worker to single threaded (`OMP_NUM_THREADS=1`, etc.) to prevent oversubscription.

---
## 5. Flickr8k Validation Mode
Run a small validation subset, producing a score dump file (`score\ttext` per line, grouped by image):
```bash
python benchmark_clip_sglang.py \
  --mode multimodal --data_source flickr8k \
  --base_url http://127.0.0.1:30000 --model openai/clip-vit-large-patch14-336 \
  --api v1 \
  --flickr_images_dir /path/to/Flicker8k_Dataset \
  --flickr_captions_file /path/to/Flickr8k.token.txt \
  --validate --validate_samples 1 --validate_group_size 5 \
  --validation_dump ./sglang_mm_validation.txt \
  --validation_distractors 2 --validate_start_group 58 \
  --warmup --warmup_iters 1
```
Key flags:
- `--validate`: enable validation logic (image paired with multiple captions & distractors).
- `--validate_group_size`: captions per image.
- `--validation_distractors`: number of randomly sampled negative captions.
- `--validation_dump`: output file path.

---
## 6. Random Synthetic Mode
For stress testing throughput without real image IO:
```bash
python benchmark_clip_sglang.py \
  --mode multimodal --data_source random \
  --base_url http://127.0.0.1:30000 --model openai/clip-vit-base-patch32 \
  --api v1 --num_samples 5000 --batch_size 128 --clip_variant base
```
Switch resolution:
```bash
--clip_variant large-336
```
Higher resolution → higher compute cost → lower throughput.

---
## 7. Output & Metrics
The benchmark prints a summary:
```
==== Summary ====
mode=multimodal, data_source=random, num_samples=10000, batch_size=100
shape=(10000, EMB_DIM)
time(s)=X.YYYY, throughput(samples/s)=ZZZZ.ZZ
```
- `throughput(samples/s)`: total samples / elapsed wall time.
- For validation runs you also get a dumped file with scores per caption.

Logs per worker (launcher mode) are written to `logs_dir/worker_*.out` and `worker_*.err`.

---
## 8. Performance Tuning
CPU:
```bash
numactl -C 0-31 -m 0 python benchmark_clip_sglang.py ...
```
- Pin processes to separate cores (`--cores`).
- Use BF16 + IPEX: `--device cpu --use-ipex True --amp bf16 --amx on` (for supported CPUs).
- Reduce memory pressure by lowering batch size if encountering slowdowns.

GPU:
- Control GPU selection: `CUDA_VISIBLE_DEVICES=0`.
- Try `--amp bf16` or `--amp fp16` for faster kernel execution.

Network / Remote:
- Prefer `data-url` for small batches; large batches may benefit from path-based loading if server can access shared storage.
- Increase `--timeout` for slower networks.

Multi-process:
- Ensure each worker uses single threads inside PyTorch / BLAS (already enforced).
- Avoid oversubscribing memory bandwidth—monitor system metrics.

---
## 9. Troubleshooting
| Issue | Hint |
|-------|------|
| Slow throughput | Lower resolution (`clip_variant base`), reduce batch size, pin fewer workers. |
| OOM (GPU) | Reduce batch size; switch to base variant. |
| Server 4xx/5xx | Check `--api` matches server endpoints; verify auth token for `openai` mode. |
| Validation dump empty | Ensure `--validate` plus existing Flickr8k paths, correct group start index. |
| Affinity not applied | Use `--use-taskset` if `sched_setaffinity` unsupported. |
| Images missing in Flickr8k | Confirm dataset directory + token file alignment. |
| BF16 unsupported | Fallback to `--amp fp16` or `--amp off`. |

---
## 10. Extending & Exporting Embeddings
Local runs can dump embeddings to `.pt` files:
```bash
--dump-txt-emb runs/clip_text_emb.pt --dump-img-emb runs/clip_img_emb.pt
```
You can later load them:
```python
import torch
text_emb = torch.load("runs/clip_text_emb.pt")
img_emb = torch.load("runs/clip_img_emb.pt")
```
To integrate a new multimodal backend, implement a class with an `encode` method under `src/backends/` and register it in `main.py`.

---
## 11. Example Batch Sweep Script
From `run_sglang_clip_cpu.sh` (simplified):
```bash
for BATCH_SIZE in 1 2 4 8 16 32 64 100 128; do
  python benchmark_clip_sglang_launcher.py --workers 2 --cores 16-17 \
    --script benchmark_clip_sglang.py --logs_dir logs_batch_${BATCH_SIZE} -- \
    --base_url=http://127.0.0.1:30000 --model=openai/clip-vit-base-patch32 \
    --api=v1 --mode=multimodal --data_source=random --image_transport=data-url \
    --num_samples=10000 --batch_size=$BATCH_SIZE --clip_variant base
done
```
Use results to plot `batch_size` vs `throughput(samples/s)`.

---
## 12. Validation Run Example
From `run_sglang_clip_val.sh` (adapt path):
```bash
python benchmark_clip_sglang.py \
  --mode multimodal --data_source flickr8k \
  --base_url=http://127.0.0.1:30000 --model=openai/clip-vit-base-patch32 \
  --api=v1 \
  --flickr_images_dir /path/to/Flicker8k_Dataset \
  --flickr_captions_file /path/to/Flickr8k.token.txt \
  --validate --validate_samples 1 --validate_group_size 5 \
  --validation_dump ./sglang_mm_validation.txt \
  --validation_distractors 2 --validate_start_group 58 \
  --warmup --warmup_iters 1
```

---
## 13. Notes
- Ensure the SGLang server is launched with CLIP or multimodal support. `benchmark_clip_sglang.py` assumes the server can accept image+text payloads under the chosen API mode.
- Synthetic image generation uses NumPy random arrays; adjust variant for realism vs speed.
- Logs may contain large JSON payloads; disable debug with `--debug` omitted for cleaner output.

