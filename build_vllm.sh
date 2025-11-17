
pip uninstall -y vllm

# 2) 安装 PyTorch CPU 版（确保 oneDNN/AMX 路径）
# pip install --upgrade pip
# pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio

# 3) 从源码装 vLLM 的 CPU 目标（关键在这个环境变量）
[ -d vllm/.git ] || git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements/cpu-build.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu

pip install -r requirements/cpu.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
  
VLLM_TARGET_DEVICE=cpu python setup.py install  # 或 VLLM_TARGET_DEVICE=cpu pip install -e .
