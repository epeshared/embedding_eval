from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

vision_process, port = launch_server_cmd(
    """
python3 -m sglang.launch_server --model-path openai/clip-vit-large-patch14-336 --tokenizer-path openai/clip-vit-large-patch14-336 --trust-remote-code --disable-overlap-schedule --is-embedding --device cpu --host 0.0.0.0 --port 30000 --skip-server-warmup --tp 1 --enable-torch-compile --torch-compile-max-bs 16 --attention-backend intel_amx --log-level info --mem-fraction-static 0.2
"""
)

wait_for_server(f"http://localhost:{port}")
