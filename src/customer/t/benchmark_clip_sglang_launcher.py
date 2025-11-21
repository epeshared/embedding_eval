#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launch multiple benchmar_clip_sglang.py processes and pin each to one CPU core.

Usage examples:
  - 启 4 个进程，自动绑到核 0..3：
    python launch_pinned_bench.py --workers 4 -- \
      --base_url http://127.0.0.1:30000 --model gme-qwen2-vl --api v1 \
      --mode multimodal --data_source random --num_samples 10000 --batch_size 100

  - 显式选择核心并使用 taskset：
    python launch_pinned_bench.py --workers 3 --cores 0,2,4 --use-taskset --
      --base_url http://127.0.0.1:30000 --model Qwen/Qwen3-Embedding-4B --api v1 \
      --mode text --data_source random --num_samples 6000 --batch_size 200
"""

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Set

def parse_cores(s: str) -> List[int]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            a, b = int(a), int(b)
            out.extend(range(min(a, b), max(a, b) + 1))
        else:
            out.append(int(tok))
    return out

def split_cores_for_workers(cores: List[int], num_workers: int) -> List[List[int]]:
    """
    将 cores 平均分配给 num_workers 个进程
    返回: [[cores_for_worker_0], [cores_for_worker_1], ...]
    
    例如: cores=[40..79], workers=2 -> [[40..59], [60..79]]
    """
    total = len(cores)
    cores_per_worker = total // num_workers
    remainder = total % num_workers
    
    result = []
    start = 0
    for i in range(num_workers):
        # 余数分配给前几个 worker
        size = cores_per_worker + (1 if i < remainder else 0)
        result.append(cores[start:start + size])
        start += size
    return result

def set_common_thread_env(env: dict) -> None:
    # 避免每个子进程内部再使用多线程导致超额并行
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    # oneDNN/ipex 有时也识别 OMP 设置；如需更激进可加：
    # env.setdefault("ONEDNN_PRIMITIVE_CACHE_CAPACITY", "1024")

def build_python_cmd(python_exe: str, script_path: str, passthrough_args: List[str]) -> List[str]:
    return [python_exe, script_path, *passthrough_args]

def start_proc_taskset(cores: List[int], cmd: List[str], env: dict, log_out, log_err):
    # 用 taskset 绑定到多个核心
    core_list = ",".join(map(str, cores))
    full = ["taskset", "-c", core_list, *cmd]
    return subprocess.Popen(full, env=env, stdout=log_out, stderr=log_err, preexec_fn=os.setsid)

def start_proc_affinity(cores: List[int], cmd: List[str], env: dict, log_out, log_err):
    # 用 sched_setaffinity 绑定到多个核心
    def preexec():
        try:
            os.sched_setaffinity(0, set(cores))
        except AttributeError:
            # 某些系统/环境不支持，退回到不绑定（由调用方用 --use-taskset 解决）
            pass
        os.setsid()
    return subprocess.Popen(cmd, env=env, stdout=log_out, stderr=log_err, preexec_fn=preexec)

def main():
    ap = argparse.ArgumentParser(description="Launch pinned benchmar_clip_sglang.py workers")
    ap.add_argument("--python", default=sys.executable, help="Python executable to use")
    ap.add_argument("--script", default="benchmark_clip_sglang.py", help="Benchmark script path")
    ap.add_argument("--workers", type=int, required=True, help="Number of worker processes")
    ap.add_argument("--cores", type=str, default="", help="Core list, e.g. '0,2,4-7'. If empty, use [0..workers-1]")
    ap.add_argument("--logs_dir", type=str, default="logs_pinned", help="Directory to store per-worker logs")
    ap.add_argument("--stagger_sec", type=float, default=0.0, help="Stagger launch interval (seconds)")
    ap.add_argument("--use-taskset", action="store_true", help="Use `taskset -c` instead of sched_setaffinity")
    ap.add_argument("--dry-run", action="store_true", help="Print the commands and exit")
    ap.add_argument("dashdash", nargs="?", default=None)
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="Arguments after -- are passed to the benchmark script verbatim")
    args = ap.parse_args()

    # 解析 passthrough 参数（去掉开头的 --）
    passthrough = args.rest
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    # 解析核心列表
    if args.cores:
        cores = parse_cores(args.cores)
    else:
        cores = list(range(args.workers))
    
    # 将核心分配给每个 worker
    cores_per_worker = split_cores_for_workers(cores, args.workers)
    
    if len(cores) < args.workers:
        print(f"[Warning] total cores({len(cores)}) < workers({args.workers}). Workers will share cores.")
    
    print(f"[Launcher] Core allocation:")
    for i, worker_cores in enumerate(cores_per_worker):
        print(f"  worker-{i}: cores {worker_cores[0]}-{worker_cores[-1]} ({len(worker_cores)} cores)")

    # 路径与环境
    script_path = str(Path(args.script).resolve())
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    base_env = os.environ.copy()
    set_common_thread_env(base_env)

    # 生成命令
    cmd_base = build_python_cmd(args.python, script_path, passthrough)

    if args.dry_run:
        for i in range(args.workers):
            worker_cores = cores_per_worker[i]
            core_str = ",".join(map(str, worker_cores))
            if args.use_taskset:
                full = ["taskset", "-c", core_str, *cmd_base]
            else:
                full = cmd_base
            print(f"[dry-run] worker-{i} cores={core_str} cmd: {shlex.join(full)}  env: OMP/MKL/OPENBLAS=1")
        return

    procs = []
    log_files = []
    print(f"[Launcher] starting {args.workers} workers; total_cores={len(cores)}; logs={logs_dir}")

    try:
        for i in range(args.workers):
            worker_cores = cores_per_worker[i]
            log_out = open(logs_dir / f"worker_{i}.out", "w", buffering=1)
            log_err = open(logs_dir / f"worker_{i}.err", "w", buffering=1)
            log_files.extend([log_out, log_err])

            if args.use_taskset:
                p = start_proc_taskset(worker_cores, cmd_base, base_env, log_out, log_err)
            else:
                p = start_proc_affinity(worker_cores, cmd_base, base_env, log_out, log_err)

            procs.append((i, worker_cores, p))
            core_range = f"{worker_cores[0]}-{worker_cores[-1]}" if len(worker_cores) > 1 else str(worker_cores[0])
            print(f"[Launcher] worker-{i} PID={p.pid} pinned to cores {core_range}; log={log_out.name}")

            if args.stagger_sec > 0 and i < args.workers - 1:
                time.sleep(args.stagger_sec)

        # 等待子进程完成；同时处理 Ctrl-C
        def terminate_all(sig_name="SIGINT"):
            print(f"\n[Launcher] {sig_name} received. Terminating all workers...")
            for i, worker_cores, p in procs:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except Exception:
                    pass

        try:
            exit_codes = {}
            while procs:
                i, worker_cores, p = procs[0]
                ret = p.wait()
                exit_codes[i] = ret
                core_range = f"{worker_cores[0]}-{worker_cores[-1]}" if len(worker_cores) > 1 else str(worker_cores[0])
                print(f"[Launcher] worker-{i} (PID={p.pid}, cores={core_range}) exited with code {ret}")
                procs.pop(0)
            print("[Launcher] all workers finished.")
        except KeyboardInterrupt:
            terminate_all("KeyboardInterrupt")
    finally:
        for f in log_files:
            try:
                f.close()
            except Exception:
                pass

if __name__ == "__main__":
    main()
