#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, List
from transformers import AutoProcessor
from optimum.intel.openvino import OVModelForFeatureExtraction

# ---------- 小工具 ----------

# 第一次：从 HF 权重导出为 OpenVINO IR（显式指定 library）
hf_id = "/home/xtang/vdb-sandbox/models/embedding/models/openai/clip-vit-base-patch32"
ov_dir = Path("./ov_models/clip-vit-base-patch32")
device_ov = "CPU"

ov_model = OVModelForFeatureExtraction.from_pretrained(
     hf_id,
     export=True,
     library="transformers",          # 关键：告诉它来自 transformers
     task="feature-extraction",       # 关键：CLIP走特征提取任务
     device="CPU",                    # 可选
 )
ov_model.save_pretrained(ov_dir)
