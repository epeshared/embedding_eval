export CONDA_PREFIX="$(python -c 'import sys,os; print(os.environ.get("CONDA_PREFIX") or os.path.dirname(os.path.dirname(sys.executable)))')"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export HF_ENDPOINT="https://hf-mirror.com"
