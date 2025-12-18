# Scripts

## bench_flickr8k_sglang.sh

Batch benchmark for Flickr8k text/image embeddings via `main.py`.

### Usage

```bash
./bench_flickr8k_sglang.sh [online|offline]
# MODE defaults to offline if omitted
```

### Common environment overrides

- `BATCH_LIST_STR`: batch sizes list, e.g. "1 2 4 8 16" (default "100")
- `MAX_SAMPLES`: cap number of images (and captions) (default 1000)
- `CAPTIONS_PER_IMAGE`: captions per image to embed (default 1; Flickr8k has 5)
- `FLICKR8K_MODALITY`: `both` | `text` | `image` (default both)
- `WARMUP`: warmup samples; `<=0` uses default (Yahoo/Flickr behavior in code) (default -1)
- `PROFILE`: set `1`/`true` to enable profiling hooks (best-effort) (default 0)
- `FLICKR8K_IMAGES_DIR`: path to `Flicker8k_Dataset` (default repo path)
- `FLICKR8K_CAPTIONS_FILE`: path to `Flickr8k.token.txt` (default repo path)

### Online-only env

- `MODEL_PATH`: model ID (default `openai/clip-vit-base-patch32`)
- `SGL_URL`: sglang server URL (default `http://127.0.0.1:30000`)
- `SGL_API`: `v1`|`native`|`openai` (image embedding needs v1/openai) (default v1)
- `SGL_API_KEY`: API key if needed

### Offline-only env

- `MODEL_DIR`: local model path (default `/home/xtang/models/openai/clip-vit-base-patch32/`)
- `DEVICE`: `cpu`|`cuda` (default cpu)

### Examples

- Default offline run (batch 100):

  ```bash
  ./bench_flickr8k_sglang.sh
  ```

- Online text-only, batch sizes 32/64, warmup 200, enable profile:

  ```bash
  FLICKR8K_MODALITY=text \
  BATCH_LIST_STR="32 64" \
  WARMUP=200 \
  PROFILE=1 \
  ./bench_flickr8k_sglang.sh online
  ```

- Offline image-only on GPU, limit 500 images:

  ```bash
  FLICKR8K_MODALITY=image \
  DEVICE=cuda \
  MAX_SAMPLES=500 \
  ./bench_flickr8k_sglang.sh offline
  ```

## Verify Flickr8k embeddings (correctness)

`bench_flickr8k_sglang.sh` focuses on throughput and writes CSV/JSONL timing records.
To *validate that the produced image/text embeddings are correct*, use the verification scripts below.

### Recommended: one-shot run + verify (dump .pt + Recall@K)

Use `run_and_verify_flickr8k_sglang.sh` to:

1) run `main.py` with `--dump-img-emb/--dump-txt-emb` to save embeddings to `runs/*.pt`
2) run retrieval verification (sanity stats + Recall@K)

Examples:

- Offline (Engine) embeddings + verify:

  ```bash
  BATCH_SIZE=64 MAX_SAMPLES=200 CAPTIONS_PER_IMAGE=1 \
  ./scripts/run_and_verify_flickr8k_sglang.sh offline
  ```

- Online (HTTP server) embeddings + verify:

  ```bash
  BATCH_SIZE=64 MAX_SAMPLES=200 CAPTIONS_PER_IMAGE=1 \
  ./scripts/run_and_verify_flickr8k_sglang.sh online
  ```

Notes:

- Keep `CAPTIONS_PER_IMAGE` consistent across generation and verification.
- If you want Flickr8k's full setting, set `CAPTIONS_PER_IMAGE=5`.
- For offline image embedding, ensure `sglang-offline` is actually used (some setups may fall back to the local `clip` backend in `bench_flickr8k_sglang.sh`).

### Verify existing .pt files (sanity + Recall@K)

If you already have `.pt` dumps from `main.py`, run `verify_flickr8k_embeddings.sh`:

```bash
./scripts/verify_flickr8k_embeddings.sh \
  --img-pt runs/flickr8k_offline_cpu_bs64_n200_cpi1_img.pt \
  --txt-pt runs/flickr8k_offline_cpu_bs64_n200_cpi1_txt.pt \
  --captions-per-image 1 \
  --k "1,5,10" \
  --max-n 200
```

This prints:

- NaN/Inf checks and embedding norm statistics
- `paired_sim` sanity (paired text-image similarity should exceed overall average)
- Retrieval `Recall@K` for **Text→Image** and **Image→Text**

### Inspect dump files (quick sanity)

To quickly inspect tensor shapes, norms, and a few rows:

```bash
python ./scripts/inspect_pt.py runs/flickr8k_offline_cpu_bs64_n200_cpi1_img.pt --cols 10 --precision 6
python ./scripts/inspect_pt.py runs/flickr8k_offline_cpu_bs64_n200_cpi1_txt.pt --cols 10 --precision 6
```
