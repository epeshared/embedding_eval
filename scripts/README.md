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
