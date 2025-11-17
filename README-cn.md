# Embedding Evaluation Suite （向量表示评测套件）

一个模块化的 Embedding（向量表示）评测工具，支持多后端、数据集与多种运行环境（CPU / GPU / AMX / 远程服务）。设计目标：结构清晰、易扩展、可复用、脚本化友好。

---
## 目录结构概览

```
embedding_eval/
├── main.py                  # 命令行入口（参数兼容旧脚本）
├── README.md                # 原英文/或综合说明
├── README-en.md             # 英文版快速说明
├── README-cn.md             # 中文版（本文件）
├── requirements-cpu.txt     # CPU 基础依赖
├── requirements-cuda.txt    # CUDA/GPU 相关额外依赖
├── requirements-ipex.txt    # Intel IPEX / oneDNN 加速相关
├── src/
│   ├── utils.py             # pooling、归一化、CSV/JSONL、AMX 环境设置
│   ├── datasets.py          # 远程 & 本地数据集加载 + 自动列名检测
│   ├── evals.py             # 文本对相似度评测 & Food-101 zero-shot
│   ├── mteb_bridge.py       # MTEB 任务适配封装
│   └── backends/
│       ├── base.py          # BaseEncoder 接口定义
│       ├── transformers_backend.py
│       ├── vllm_backend.py
│       ├── vllm_openai_backend.py   # 通过 OpenAI 接口风格调用 vLLM
│       ├── llamacpp_backend.py
│       ├── sglang_backend.py
│       └── clip_backend.py
├── scripts/
│   ├── bench_transformer.sh # Transformers 后端基准测试
│   ├── bench_vllm_cuda.sh   # vLLM CUDA 基准测试
│   ├── bench_vllm_openai.sh # vLLM OpenAI 兼容接口测试
│   ├── bench_sglang.sh      # SGLang 基准脚本
│   ├── start_sglang_server.sh
│   ├── start_vllm_server.sh
│   └── runs/                # 运行输出示例
└── models/ / runs/ / logs/  # 本地模型与结果
```

---
## 支持的后端（Backends）
- Transformers（Hugging Face，支持 IPEX / AMP / CPU / GPU）
- vLLM（GPU 或 CPU，兼容 embedding 接口）
- llama.cpp（GGUF，本地轻量推理，支持不同 AMX / 非 AMX 构建）
- SGLang（远程服务：native `/encode` / OpenAI 风格 `/v1/embeddings` / openai SDK）
- CLIP（文本 + 图像 Zero-shot 分类，用于如 Food-101）

---
## 主要特性
- 文本对相似度评测：LCQMC / AFQMC / PAWSX-zh 等（或自定义本地数据集）
- Food-101 Zero-shot（可指定 prompt 模板）
- MTEB 快速桥接：按语言/类型自动筛选任务
- AMX / oneDNN / IPEX CPU 加速策略快速切换
- 多种批处理与服务调用模式（本地 vs 远程）
- 统一输出格式（CSV / JSONL）方便后续分析、汇总、可视化

---
## 安装

建议 Python 3.10+。

### 基础依赖（CPU 通用）
```bash
pip install -r requirements-cpu.txt
```

### GPU 相关 & IPEX / oneDNN 加速
```bash
pip install -r requirements-cuda.txt
pip install -r requirements-ipex.txt
```

### 可选组件（按需安装）
```bash
pip install vllm              # vLLM 后端
pip install llama-cpp-python  # llama.cpp Python 绑定
pip install mteb              # 运行 MTEB 评测
pip install openai            # 使用 openai SDK 调 SGLang /v1
pip install intel-extension-for-pytorch  # IPEX 加速
pip install pillow            # CLIP 需要图像处理
```

如使用 Hugging Face 国内镜像：设置 `HF_ENDPOINT` 环境变量或通过 `--hf-endpoint` 参数传入。

---
## 快速开始示例
在项目根目录执行以下命令。

### 1. Transformers（CPU + IPEX + AMX）
```bash
python main.py --backend transformers --model BAAI/bge-large-zh-v1.5 \
  --device cpu --use-ipex True --amp bf16 --amx on \
  --datasets LCQMC --batch-size 16 \
  --output-csv runs/tf_ipex.csv
```

### 2. vLLM（GPU / CPU）
```bash
# GPU 示例
CUDA_VISIBLE_DEVICES=0 python main.py --backend vllm --model Qwen/Qwen3-Embedding-4B \
  --vllm-device cuda --vllm-dtype auto --datasets LCQMC --batch-size 16

# CPU 示例（开启 AMX）
python main.py --backend vllm --model intfloat/e5-mistral-7b-instruct \
  --vllm-device cpu --vllm-dtype bfloat16 --amx on \
  --datasets LCQMC --batch-size 16
```

### 3. llama.cpp（GGUF）
```bash
python main.py --backend llamacpp --model models/bge-large-zh-v1.5.gguf \
  --llama-n-threads 16 --datasets LCQMC --batch-size 32
```
AMX / 非 AMX 构建切换：
```bash
python main.py --backend llamacpp --model models/xxx.gguf --amx on  \
  --llama-lib-amx /path/to/libllama_amx.so
python main.py --backend llamacpp --model models/xxx.gguf --amx off \
  --llama-lib-noamx /path/to/libllama_noamx.so
```

### 4. SGLang 服务端
先启动服务（参考 `scripts/start_sglang_server.sh`），再运行：
```bash
python main.py --backend sglang --model Qwen/Qwen3-Embedding-4B \
  --sgl-url http://127.0.0.1:30000 --sgl-api v1 \
  --datasets LCQMC --batch-size 32
```
其它模式：
```bash
# native /encode
python main.py --backend sglang --model Qwen/Qwen3-Embedding-4B \
  --sgl-url http://127.0.0.1:30000 --sgl-api native --datasets LCQMC

# openai SDK
python main.py --backend sglang --model Qwen/Qwen3-Embedding-4B \
  --sgl-url http://127.0.0.1:30000 --sgl-api openai --sgl-api-key sk-xxx \
  --datasets LCQMC
```

### 5. CLIP Zero-shot（Food-101）
```bash
python main.py --backend clip --model openai/clip-vit-base-patch32 \
  --datasets food101 --batch-size 64 \
  --clip-prompt "a photo of a {}"
```
可选：导出嵌入向量
```bash
... --dump-txt-emb runs/clip_text_emb.pt --dump-img-emb runs/clip_img_emb.pt
```

---
## 脚本（`scripts/`）
| 脚本 | 用途 |
|------|------|
| `bench_transformer.sh` | Transformers 后端批量/重复基准测试 |
| `bench_vllm_cuda.sh` | vLLM GPU 基准 |
| `bench_vllm_openai.sh` | vLLM 通过 OpenAI 风格接口测试 |
| `bench_sglang.sh` | SGLang embeddings 基准 |
| `start_sglang_server.sh` | 启动 SGLang 服务端 |
| `start_vllm_server.sh` | 启动 vLLM 服务端 |

可根据需要修改脚本中的模型、batch、设备参数进行批量实验。

---
## 核心 CLI 参数速览

### 通用
- `--backend`：选择后端（transformers / llamacpp / vllm / clip / sglang）
- `--model`：模型名称或路径
- `--datasets`：逗号分隔数据集：如 `LCQMC,AFQMC` 或本地文件/目录
- `--split`：使用的数据集划分（默认 `validation`）
- `--batch-size`：批大小
- `--max-samples`：限制样本数量（`-1`为全量）
- `--output-csv` / `--output-jsonl`：结果追加保存

### AMX / CPU 加速
- `--amx`：`auto|on|off`
- `--amx-verbose`：是否打印 oneDNN 详细日志

### Transformers / CLIP 特定
- `--device`：`cpu|cuda`
- `--use-ipex`：是否启用 IPEX（只在 CPU 有效）
- `--amp`：`off|auto|fp16|bf16`
- `--max-length`：tokenizer 截断长度
- `--offline`：本地离线模式（不访问网络）
- `--hf-endpoint`：自定义镜像地址
- `--trust-remote-code`：允许加载含自定义代码的模型

### llama.cpp
- `--llama-n-threads` / `--llama-n-gpu-layers` / `--llama-verbose`
- `--llama-lib-amx` / `--llama-lib-noamx`：指定不同编译的动态库

### vLLM
- `--vllm-dtype`：`auto|float32|bfloat16|float16|...`
- `--vllm-device`：`cpu|cuda`
- `--vllm-tp`：Tensor Parallel 数量
- `--vllm-max-model-len`：KV Cache 上限
- `--vllm-gpu-mem-util`：GPU 显存利用率（默认 `0.90`）

### SGLang
- `--sgl-url`：服务端基础 URL
- `--sgl-api`：`native|v1|openai`
- `--sgl-api-key`：鉴权 Token（若服务器需要）
- `--profile` / `--profile-steps` / `--profile-output-dir`：远程性能分析控制

### MTEB
- `--mteb`：是否运行 MTEB
- `--mteb-tasks`：指定任务列表（为空则根据语言/类型自动）
- `--mteb-task-langs` / `--mteb-task-types`
- `--mteb-output-dir`：MTEB 结果输出目录

---
## 数据集加载逻辑
- 远程映射：`LCQMC -> C-MTEB/LCQMC`，`AFQMC -> clue/afqmc`，`pawsx-zh` 等映射到 `paws-x`。
- 本地：
  - 传目录：优先 `load_dataset(path)`，失败则 `load_from_disk(path)`，再向下递归查找文件。
  - 传文件：支持 `csv/tsv/json/jsonl/parquet`。
- 自动列名检测（大小写不敏感）：
  - 文本对：`sentence1/sentence2`，`text1/text2`，`question1/question2`，`q/p`，`s1/s2`，`query/passage` 等。
  - 标签：`label`，`score`，`target`，`y`，`similar`。数值型 >=0.5 判定为正；布尔转 0/1；字符串支持 `1/0/true/false/yes/no/y/n`。

---
## 输出格式示例
每个数据集评测会追加一条记录（CSV 行或 JSON 行）：
```json
{
  "dataset": "LCQMC",
  "split": "validation",
  "n_samples": 12345,
  "acc": 0.8621,
  "f1": 0.8590,
  "encode_time": 12.345678,
  "score_time": 0.012345,
  "total_time": 12.358023,
  "qps": 999.99
}
```
可用于后续汇总/绘图/对比。

---
## AMX 与 oneDNN / IPEX 策略
- Transformers / CLIP（CPU + IPEX）：
  - `--amx on` → `ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX`
  - `--amx off` → `ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16`
  - `--amx auto` → 让 oneDNN 自动选择
  - `--amx-verbose True` 输出详细 Kernel 使用
- vLLM（CPU 构建）同样设置上述环境变量。
- llama.cpp：通过不同编译库选择 AMX / 非 AMX。

注意：需要硬件支持（如 Intel Sapphire Rapids / Granite Rapids）及操作系统内核支持。

---
## MTEB 使用示例
```bash
python main.py --backend transformers --model BAAI/bge-large-zh-v1.5 \
  --device cpu --use-ipex True --amp bf16 --amx on \
  --mteb True --mteb-task-langs zh --mteb-output-dir runs/mteb
```
自动任务预设：
- `zh`：`[AFQMC, LCQMC, BQ, PAWSX-zh, ATEC]`
- `multi`：`[STS17, STS22]`
- 英文默认 STS 系列：`[STS12, STS13, STS14, STS15, STS16, STSBenchmark, SICK-R]`

---
## 性能与调优建议
```bash
# 绑定 CPU 核心与内存节点提升稳定性
numactl -C 0-31 -m 0 python main.py ...
```
- AMX 机器：优先 `--amp bf16` + IPEX 获得融合算子收益。
- GPU：使用 `CUDA_VISIBLE_DEVICES` 控制设备；必要时限制显存或设置并行度。
- 大批量任务：监控 QPS / encode_time，适当调整 `--batch-size`。

---
## 扩展新后端步骤
1. 新建文件：`src/backends/my_backend.py`。
2. 实现统一接口：
   ```python
   class MyEncoder(BaseEncoder):
       @torch.inference_mode()
       def encode(self, texts: List[str], batch_size: int = 128) -> torch.Tensor:
           ...
   ```
3. 在 `main.py` 中增加对应的 `elif` 分支解析命令行参数并实例化。
4. （可选）添加脚本到 `scripts/` 进行基准测试。

---
## 常见问题（Troubleshooting）
| 问题 | 排查建议 |
|------|----------|
| 模型下载失败 | 使用 `--offline True` + 先行缓存；或设置镜像 `HF_ENDPOINT` |
| vLLM device 参数报错 | 某些版本弃用 `device=`，代码中已做兼容回退 |
| SGLang `/v1/embeddings` 失败 | 确认服务端开放该路径；鉴权时传入 `--sgl-api-key` |
| AMX 未生效 | 确认 CPU 支持、IPEX 已安装、设置了 `--amx on` 并使用 BF16 |
| 本地数据集列名无法识别 | 检查字段是否在支持别名列表内；JSON/JSONL 顶层需直接包含字段 |
| CLIP 图像处理报错 | 安装 `pillow` |


