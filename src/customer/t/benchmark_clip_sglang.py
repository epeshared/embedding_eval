#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import io
import os
import base64
import time
import json
from typing import List, Union, Optional, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
from PIL import Image
import requests


# ============================================================
# Utils
# ============================================================
def np_to_png_bytes(arr: np.ndarray) -> bytes:
    """HWC uint8 (RGB) -> PNG bytes"""
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def png_bytes_to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def read_flickr8k_captions(captions_file: str) -> Dict[str, List[str]]:
    """Flickr8k.token.txt -> {filename: [captions...]}"""
    mp: Dict[str, List[str]] = {}
    with open(captions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            left, cap = line.split("\t", 1)
            img, _ = left.split("#", 1)
            mp.setdefault(img, []).append(cap)
    return mp


def load_flickr8k_pairs(
    images_dir: str,
    captions_file: str,
    total: int,
    pick_caption: str = "first",
) -> Tuple[List[np.ndarray], List[str]]:
    """1 image ↔ 1 caption, total samples."""
    cap_map = read_flickr8k_captions(captions_file)
    filenames = [
        fn for fn in os.listdir(images_dir)
        if fn.lower().endswith((".jpg", ".jpeg", ".png")) and fn in cap_map
    ]
    if not filenames:
        raise RuntimeError(f"{images_dir} has no images matching {os.path.basename(captions_file)}")
    filenames.sort()

    rng = np.random.default_rng()
    images, texts = [], []
    n = len(filenames)

    for i in range(total):
        fn = filenames[i % n]
        with Image.open(os.path.join(images_dir, fn)) as im:
            im = im.convert("RGB")
            arr = np.array(im, dtype=np.uint8)
        caps = cap_map[fn]
        txt = caps[rng.integers(0, len(caps))] if pick_caption == "random" else caps[0]
        images.append(arr)
        texts.append(txt)

    return images, texts


def load_flickr8k_groups_with_names(
    images_dir: str,
    captions_file: str,
    total_groups: int,
    group_size: int = 5,
    start_group: int = 0,
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    验证分组展开：每组 1 张图 + group_size 条 caption。
    展开后 images 与 texts 等长： [img,c1,c2,...,img,c1,c2,...]
    """
    cap_map = read_flickr8k_captions(captions_file)
    filenames = [
        fn for fn in os.listdir(images_dir)
        if fn.lower().endswith((".jpg", ".jpeg", ".png")) and fn in cap_map
    ]
    if not filenames:
        raise RuntimeError(f"{images_dir} has no images matching {os.path.basename(captions_file)}")
    filenames.sort()

    if len(filenames) > 0:
        start_group = max(0, start_group) % len(filenames)
    else:
        start_group = 0

    images: List[np.ndarray] = []
    texts: List[str] = []
    group_filenames: List[str] = []

    for gi in range(total_groups):
        fn = filenames[(start_group + gi) % len(filenames)]
        with Image.open(os.path.join(images_dir, fn)) as im:
            im = im.convert("RGB")
            arr = np.array(im, dtype=np.uint8)

        caps = cap_map[fn]
        if len(caps) < group_size:
            caps = caps + caps[: (group_size - len(caps))]
        else:
            caps = caps[:group_size]

        group_filenames.append(fn)

        images.append(arr)
        texts.append(caps[0])
        for c in caps[1:group_size]:
            images.append(arr)
            texts.append(c)

    return images, texts, group_filenames


def make_random_images_and_texts(total: int, clip_variant: str) -> Tuple[List[np.ndarray], List[str]]:
    if clip_variant == "large-336":
        h, w = 336, 336
    else:
        h, w = 224, 224
    images = [
        np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        for _ in range(total)
    ]
    texts = ["a beautiful landscape"] * total
    return images, texts


def make_random_texts(total: int) -> List[str]:
    return ["a beautiful landscape"] * total


# ============================================================
# Safe payload print (hide image content)
# ============================================================
def print_payload_safe(payload: dict, title: str = "PAYLOAD", enable: bool = True):
    """打印 payload，但隐藏 image 的具体值（如 data-url / base64 / 路径）"""
    if not enable:
        return
    import copy
    safe = copy.deepcopy(payload)

    def _mask(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "image":
                    obj[k] = "[image omitted]"
                else:
                    _mask(v)
        elif isinstance(obj, list):
            for x in obj:
                _mask(x)

    _mask(safe)
    print(f"\n=== {title} ===")
    print(json.dumps(safe, indent=2, ensure_ascii=False))


# ============================================================
# Preprocessing
# ============================================================
def normalize_image_repr_one(img: Union[str, bytes, np.ndarray], image_transport: str) -> str:
    if image_transport == "data-url":
        if isinstance(img, bytes):
            return png_bytes_to_data_url(img)
        elif isinstance(img, np.ndarray):
            return png_bytes_to_data_url(np_to_png_bytes(img))
        elif isinstance(img, str) and img.startswith("data:image/"):
            return img
        else:
            raise ValueError("image=data-url requires numpy / PNG bytes / existing data-url")
    else:
        if not isinstance(img, str) or not img:
            raise ValueError("image_transport='path/url' requires a string path or URL")
        return img


def normalize_images_bulk(
    images: List[Union[str, bytes, np.ndarray]],
    image_transport: str,
    max_workers: int = 0
) -> List[str]:
    if not images:
        return []
    if not max_workers or max_workers <= 0:
        return [normalize_image_repr_one(img, image_transport) for img in images]

    outs: List[Optional[str]] = [None] * len(images)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(normalize_image_repr_one, img, image_transport): idx
                for idx, img in enumerate(images)}
        for f in as_completed(futs):
            idx = futs[f]
            outs[idx] = f.result()
    return outs  # type: ignore


# ============================================================
# SGLangEncoder
# ============================================================
class SGLangEncoder:
    def __init__(self, base_url: str, model: str,
                 api: str = "v1", api_key: str = "", timeout: float = 120.0,
                 seed: int = 42, debug: bool = True,
                 warmup: bool = False, warmup_iters: int = 8, profile: bool = False):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api = api.lower()
        self.api_key = api_key or ""
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Connection": "close"})

        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.debug = bool(debug)
        self.warmup = bool(warmup)
        self.warmup_iters = int(max(0, warmup_iters))
        self._did_warmup = False
        self.profile = profile

        print(f"[Init:sglang] url={self.base_url}, model='{self.model}', api='{self.api}', "
              f"seed={self.seed}, debug={self.debug}, warmup={self.warmup}x{self.warmup_iters}")

        self._openai_client = None
        if self.api == "openai":
            try:
                import openai  # type: ignore
            except Exception as e:
                raise RuntimeError(f"openai package not installed: {e}")
            self._openai_client = openai.Client(
                base_url=f"{self.base_url}/v1",
                api_key=(self.api_key or "None")
            )

    # ---------------- text ----------------
    def _encode_v1_any(self, inputs: Union[str, List[str]]) -> List[List[float]]:
        """纯文本 embeddings"""
        url = f"{self.base_url}/v1/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model or "default",
            "input": inputs,
            "encoding_format": "float",
        }
        print_payload_safe(payload, "PAYLOAD /v1/embeddings (TEXT)", enable=self.debug)

        resp = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"/v1/embeddings text {resp.status_code}: {resp.text[:800]}")
        js = resp.json()
        if not isinstance(js, dict) or "data" not in js or not isinstance(js["data"], list):
            raise RuntimeError(f"Unexpected /v1/embeddings response: {js}")
        rows = sorted(js["data"], key=lambda d: d.get("index", 0))
        vecs = []
        for r in rows:
            emb = r.get("embedding", r.get("vector", None))
            if emb is None:
                raise RuntimeError(f"Missing 'embedding' in item: {r}")
            vecs.append([float(x) for x in emb])
        return vecs

    @torch.inference_mode()
    def encode_texts(self, texts: List[str], batch_size: int = 128) -> torch.Tensor:
        if not texts:
            return torch.empty(0, 0)

        out = []
        if self.api == "v1":
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                vecs = self._encode_v1_any(batch)
                out.append(torch.nn.functional.normalize(torch.tensor(vecs, dtype=torch.float32), p=2, dim=1))
            return torch.cat(out, dim=0)

        raise RuntimeError("Only v1 text is implemented in this example")

    # ---------------------------------------------------------
    # 小工具：打印 pairlist（隐藏 image）
    # ---------------------------------------------------------
    def _debug_print_pairlist(self, pairlist: List[Dict[str, str]]):
        if not self.debug:
            return
        printable = []
        for item in pairlist:
            if "image" in item:
                printable.append({"image": "[image omitted]"})
            else:
                printable.append({"text": item.get("text", "")})
        print("\n=== VALIDATION PAYLOAD (pairlist) ===")
        print(json.dumps(printable, indent=2, ensure_ascii=False))

    # ---------------------------------------------------------
    # 小工具：打印 resp.json()，把 embedding 截断
    # ---------------------------------------------------------
    def _debug_print_resp_json(self, js: Dict, enable: bool = True):
        if not (self.debug and enable):
            return

        def truncate_embeddings(obj):
            if isinstance(obj, dict):
                new_obj = {}
                for k, v in obj.items():
                    if k == "embedding" and isinstance(v, list):
                        new_obj[k] = f"... ({len(v)} floats) ..."
                    else:
                        new_obj[k] = truncate_embeddings(v)
                return new_obj
            elif isinstance(obj, list):
                return [truncate_embeddings(x) for x in obj]
            else:
                return obj

        trimmed = truncate_embeddings(js)
        print("\n=== VALIDATION RESP (trimmed) ===")
        print(json.dumps(trimmed, indent=2, ensure_ascii=False))

    # ---------------------------------------------------------
    # 单对打分：{"text": ..., {"image": ...}}
    # ---------------------------------------------------------
    def _score_image_text_pair(self, img_repr: str, txt: str, show_debug: bool = True) -> Tuple[float, Optional[Dict]]:
        """
        假设后端返回两条 embedding（index 排序后 0=text, 1=image），
        直接计算两者的余弦相似度作为分数。
        """
        url = f"{self.base_url}/v1/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model or "default",
            "input": [
                {"text": txt},
                {"image": img_repr}
            ]
        }
        print_payload_safe(payload, "PAYLOAD /v1/embeddings (PAIR)", enable=(self.debug and show_debug))

        resp = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"/v1/embeddings pair {resp.status_code}: {resp.text[:1000]}")

        js = resp.json()
        self._debug_print_resp_json(js, enable=show_debug)

        data = js.get("data", [])
        if not isinstance(data, list) or len(data) < 2:
            raise RuntimeError(f"expect 2 embeddings (text,image), got: {data}")

        try:
            rows = sorted(data, key=lambda d: d.get("index", 0))
        except Exception:
            rows = data

        emb_text = rows[0].get("embedding")
        emb_image = rows[1].get("embedding")
        if not (isinstance(emb_text, list) and isinstance(emb_image, list)):
            raise RuntimeError("missing embedding list in response items 0/1")

        v_text = torch.tensor(emb_text, dtype=torch.float32).unsqueeze(0)
        v_image = torch.tensor(emb_image, dtype=torch.float32).unsqueeze(0)
        score = torch.cosine_similarity(v_text, v_image, dim=1).item()
        print(f"[Pair Score] text='{txt[:30]}...', score={score:.6f}")
        return float(score), js

    # ---------------------------------------------------------
    # 预热：对给定 (img, text) 调用 N 次，不写文件，不打印 debug
    # ---------------------------------------------------------
    def _do_warmup(self, img_repr: str, txt: str):
        if not (self.warmup and self.warmup_iters > 0):
            return
        t0 = time.time()
        for _ in range(self.warmup_iters):
            try:
                # show_debug=False：不打印 payload/resp
                self._score_image_text_pair(img_repr, txt, show_debug=False)
            except Exception:
                # 预热阶段出错直接忽略（不影响正式流程）
                pass
        self._did_warmup = True
        if self.debug:
            print(f"[Warmup] ran {self.warmup_iters} iters in {time.time() - t0:.4f}s")

    # ---------------------------------------------------------
    # 验证模式（单对式；组头只写一次；每条一行 score\ttext）
    # ---------------------------------------------------------
    def _encode_mm_validation(
        self,
        texts: List[str],
        image_reprs: List[str],
        validate_group_size: int,
        validate_max_samples: int,
        group_filenames: Optional[List[str]],
        validation_dump_path: Optional[str],
        global_caption_pool: Optional[List[Tuple[str, str]]],
        validation_distractors: int,
        validate_start_group: int = 0,
    ) -> torch.Tensor:
        if validate_group_size <= 0:
            validate_group_size = 5
        real_group_span = validate_group_size

        f_out = open(validation_dump_path, "w", encoding="utf-8") if validation_dump_path else None

        outs: List[torch.Tensor] = []
        total_elapsed = 0.0

        total_groups_possible = len(texts) // real_group_span
        start_group = max(0, min(validate_start_group, max(0, total_groups_possible - 1)))
        offset = start_group * real_group_span
        group_index = start_group

        groups_sent = 0
        while offset < len(texts) and groups_sent < validate_max_samples:
            remain = len(texts) - offset
            real_cnt = min(real_group_span, remain)

            sub_t = texts[offset: offset + real_cnt]
            sub_i = image_reprs[offset: offset + real_cnt]

            img_repr = sub_i[0]
            real_texts = sub_t[:]

            # === Warmup：只在正式评分前跑一次 ===
            if not self._did_warmup and self.warmup and self.warmup_iters > 0:
                # 用第一条真实 caption 做预热
                warmup_txt = real_texts[0] if len(real_texts) > 0 else "warmup"
                self._do_warmup(img_repr, warmup_txt)

            
            # === 干扰项：对当前图片组，只从非本图片的 caption 里随机抽取 ===
            distract_texts: List[str] = []
            cur_img_name = None
            if group_filenames and group_index < len(group_filenames):
                cur_img_name = group_filenames[group_index]
            if validation_distractors > 0 and global_caption_pool:
                distract_texts = self._sample_distractors(
                    need=validation_distractors,
                    global_caption_pool=global_caption_pool,
                    cur_img_name=cur_img_name,
                )

            ordered_cap_texts = real_texts + distract_texts

            # 只写一次组头
            if f_out is not None:
                img_name = (
                    group_filenames[group_index]
                    if group_filenames and group_index < len(group_filenames)
                    else f"group_{group_index}"
                )
                f_out.write(f"[GROUP {group_index}] image={img_name}\n")

            # 逐条发单对请求，边算边写一行 score \t text
            scores: List[float] = []
            for txt in ordered_cap_texts:
                t0 = time.time()
                try:
                    s, _ = self._score_image_text_pair(img_repr, txt, show_debug=True)
                except Exception as e:
                    print(f"[Group {group_index}] pair score error: {e}")
                    s = -1e9
                total_elapsed += (time.time() - t0)
                scores.append(s)
                if f_out is not None:
                    f_out.write(f"{s:.6f}\t{txt}\n")

            # 组尾空行
            if f_out is not None:
                f_out.write("\n")

            # 可选：占位张量（不参与实际评估）
            if len(scores) > 0:
                outs.append(torch.full((1, 1), float('nan')))

            offset += real_cnt
            groups_sent += 1
            group_index += 1

        if f_out is not None:
            f_out.close()

        print(f"[SGLang MM] Total encode time for validation pairs: {total_elapsed:.4f} s")
        return torch.cat(outs, dim=0) if outs else torch.empty(0, 0)


    def _sample_distractors(self,
                            need: int,
                            global_caption_pool: Optional[List[Tuple[str, str]]],
                            cur_img_name: Optional[str]) -> List[str]:
        """从全局池随机抽取 need 条干扰 caption，排除当前图片的 caption。"""
        if not global_caption_pool or need <= 0:
            return []

        # 用时间纳秒 + 进程ID + 基础seed 混合成一次性随机源，让每次运行都不同
        import os, time
        mix_seed = (self.seed ^ os.getpid() ^ (time.time_ns() & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
        rnd = np.random.default_rng(mix_seed)

        chosen: List[str] = []

        sample_size = min(len(global_caption_pool), max(need * 10, need))
        if sample_size > 0:
            idxs = rnd.choice(len(global_caption_pool), size=sample_size, replace=False)
            for idx2 in idxs:
                fn, cap = global_caption_pool[idx2]
                if cur_img_name is not None and fn == cur_img_name:
                    continue
                chosen.append(cap)
                if len(chosen) >= need:
                    break

        if len(chosen) < need:
            # 为了再多一些随机性，这里顺手打乱一下顺序再补齐
            perm = rnd.permutation(len(global_caption_pool))
            for idx2 in perm:
                fn, cap = global_caption_pool[idx2]
                if cur_img_name is not None and fn == cur_img_name:
                    continue
                chosen.append(cap)
                if len(chosen) >= need:
                    break

        return chosen[:need]



    # ---------------- multimodal (统一入口) ----------------
    def encode_v1_multimodal_pairlist_prepared(
        self,
        texts: List[str],
        image_reprs: List[str],
        batch_size: int = 32,
        do_validation: bool = False,
        validate_group_size: int = 5,
        validate_max_samples: int = 50,
        group_filenames: Optional[List[str]] = None,
        validation_dump_path: Optional[str] = None,
        global_caption_pool: Optional[List[Tuple[str, str]]] = None,
        validation_distractors: int = 0,
        validate_start_group: int = 0,
    ) -> torch.Tensor:
        """
        统一入口：
        - 验证：走 _encode_mm_validation(...)（单对式）
        - 普通：走按 batch 的 [text, image]（本脚本主要用于验证路径）
        """
        assert len(texts) == len(image_reprs), "texts and image_reprs must have the same length"
        if self.api != "v1":
            raise RuntimeError("Multimodal is implemented only via /v1/embeddings; set api='v1'")
        if not texts:
            return torch.empty(0, 0)

        if do_validation:
            return self._encode_mm_validation(
                texts=texts,
                image_reprs=image_reprs,
                validate_group_size=validate_group_size,
                validate_max_samples=validate_max_samples,
                group_filenames=group_filenames,
                validation_dump_path=validation_dump_path,
                global_caption_pool=global_caption_pool,
                validation_distractors=validation_distractors,
                validate_start_group=validate_start_group,
            )

        # 普通批处理（保留，但常用不上）
        url = f"{self.base_url}/v1/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        outs: List[torch.Tensor] = []
        total_elapsed = 0.0
        sample_len = len(texts)
        sample = (sample_len // batch_size // 2) * batch_size

        for batch_start in range(0, len(texts), batch_size):
            sub_t = texts[batch_start:batch_start + batch_size]
            sub_i = image_reprs[batch_start:batch_start + batch_size]

            input_list: List[Dict[str, str]] = []
            for img_repr, txt in zip(sub_i, sub_t):
                input_list.append({"text": txt})
                input_list.append({"image": img_repr})

            payload = {"model": self.model or "default", "input": input_list}
            print_payload_safe(payload, "PAYLOAD /v1/embeddings (MULTIMODAL BATCH)", enable=self.debug)

            t0 = time.time()
            if self.profile and batch_start == sample:
                print("Starting profiler...")
                try:
                    resp_start = self.session.post(f"{self.base_url}/start_profile",
                                                   json={"activities": ["CPU", "CUDA"], "record_shapes": True},
                                                   timeout=10)
                    if resp_start.status_code == 200:
                        print("Profiler started")
                except Exception as e:
                    print(f"Failed to start profiler: {e}")
            resp = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
            if self.profile and batch_start == sample:
                print("Stopping profiler...")
                try:
                    resp_stop = self.session.post(f"{self.base_url}/stop_profile", timeout=10)
                    if resp_stop.status_code == 200:
                        print("Profiler stopped")
                except Exception as e:
                    print(f"Failed to stop profiler: {e}")
            total_elapsed += (time.time() - t0)
            if batch_start == sample:
                print("current batch uses ", time.time() - t0, " s")
            if resp.status_code != 200:
                raise RuntimeError(f"/v1/embeddings multimodal {resp.status_code}: {resp.text[:1000]}")

            js = resp.json()
            rows = sorted(js["data"], key=lambda d: d.get("index", 0))

            # 提取文本向量（偶数位 0,2,4,... 是 text）
            text_vecs = []
            for idx, r in enumerate(rows):
                if idx % 2 == 0:
                    emb = r.get("embedding", r.get("vector", None))
                    if emb is None:
                        raise RuntimeError(f"Missing 'embedding' in item: {r}")
                    text_vecs.append([float(x) for x in emb])

            t = torch.tensor(text_vecs, dtype=torch.float32)
            t = torch.nn.functional.normalize(t, p=2, dim=1)
            outs.append(t)

        print(f"[SGLang MM] Total encode time for {len(texts)} samples: {total_elapsed:.4f} s")
        return torch.cat(outs, dim=0) if outs else torch.empty(0, 0)

    def encode_v1_multimodal_pairlist(
        self,
        texts: List[str],
        images: List[Union[str, bytes, np.ndarray]],
        batch_size: int = 32,
        image_transport: str = "data-url",
        do_validation: bool = False,
        validate_group_size: int = 5,
        validate_max_samples: int = 50,
        group_filenames: Optional[List[str]] = None,
        validation_dump_path: Optional[str] = None,
        global_caption_pool: Optional[List[Tuple[str, str]]] = None,
        validation_distractors: int = 0,
        validate_start_group: int = 0,
    ) -> torch.Tensor:
        image_reprs = normalize_images_bulk(images, image_transport)
        return self.encode_v1_multimodal_pairlist_prepared(
            texts=texts,
            image_reprs=image_reprs,
            batch_size=batch_size,
            do_validation=do_validation,
            validate_group_size=validate_group_size,
            validate_max_samples=validate_max_samples,
            group_filenames=group_filenames,
            validation_dump_path=validation_dump_path,
            global_caption_pool=global_caption_pool,
            validation_distractors=validation_distractors,
            validate_start_group=validate_start_group,
        )


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser("SGLang embeddings | text & text+image | random / Flickr8k")
    # service
    ap.add_argument("--base_url", type=str, default="http://127.0.0.1:30000")
    ap.add_argument("--model", type=str, default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--api", choices=["v1", "native", "openai"], default="v1")
    ap.add_argument("--api_key", type=str, default="")
    ap.add_argument("--timeout", type=float, default=120.0)

    # task
    ap.add_argument("--mode", choices=["text", "multimodal"], default="multimodal")
    ap.add_argument("--data_source", choices=["random", "flickr8k"], default="flickr8k")

    # random image variant
    ap.add_argument("--clip_variant", choices=["base", "large-336"], default="base")

    # flickr8k
    ap.add_argument("--flickr_images_dir", type=str, default=None)
    ap.add_argument("--flickr_captions_file", type=str, default=None)
    ap.add_argument("--flickr_caption_pick", choices=["first", "random"], default="first")

    # workload
    ap.add_argument("--num_samples", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)

    # images
    ap.add_argument("--image_transport", choices=["data-url", "path/url"], default="data-url")
    ap.add_argument("--image_prep_workers", type=int, default=8)

    # validation
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--validate_samples", type=int, default=1)
    ap.add_argument("--validate_group_size", type=int, default=5)
    ap.add_argument("--validation_dump", type=str, default="./sglang_mm_validation.txt")
    ap.add_argument("--validation_distractors", type=int, default=2)
    ap.add_argument("--validate_start_group", type=int, default=0)
    ap.add_argument("--validate_pick_file", type=str, default=None)

    # reproducibility
    ap.add_argument("--seed", type=int, default=42)

    # debug & warmup
    ap.add_argument("--debug", action="store_true", help="print payload/resp debug")
    ap.add_argument("--warmup", action="store_true", help="run warmup calls before scoring")
    ap.add_argument("--warmup_iters", type=int, default=8, help="number of warmup calls")

    # profile
    ap.add_argument("--profile", action="store_true", help="run PyTorch Profile")

    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    enc = SGLangEncoder(
        base_url=args.base_url,
        model=args.model,
        api=args.api,
        api_key=args.api_key,
        timeout=args.timeout,
        seed=args.seed,
        debug=args.debug,
        warmup=args.warmup,
        warmup_iters=args.warmup_iters,
        profile=args.profile,
    )

    if args.mode == "text":
        if args.data_source == "random":
            texts = make_random_texts(args.num_samples)
        else:
            if not args.flickr_captions_file:
                raise ValueError("--data_source flickr8k requires --flickr_captions_file")
            cap_map = read_flickr8k_captions(args.flickr_captions_file)
            all_caps = []
            rng = np.random.default_rng(args.seed)
            for _, caps in cap_map.items():
                if not caps:
                    continue
                if args.flickr_caption_pick == "first":
                    all_caps.append(caps[0])
                else:
                    all_caps.append(caps[int(rng.integers(0, len(caps)))] )
            if not all_caps:
                raise RuntimeError("Flickr8k captions are empty")
            texts = [all_caps[i % len(all_caps)] for i in range(args.num_samples)]

        t0 = time.time()
        embs = enc.encode_texts(texts, batch_size=args.batch_size)

    else:
        if args.data_source == "random":
            images, texts = make_random_images_and_texts(args.num_samples, args.clip_variant)
            image_reprs = normalize_images_bulk(
                images, args.image_transport, max_workers=args.image_prep_workers
            )
            t0 = time.time()
            embs = enc.encode_v1_multimodal_pairlist_prepared(
                texts=texts,
                image_reprs=image_reprs,
                batch_size=args.batch_size,
                do_validation=False,
            )
        else:
            if not args.flickr_images_dir or not args.flickr_captions_file:
                raise ValueError("--data_source flickr8k requires --flickr_images_dir and --flickr_captions_file")

            cap_map_all = read_flickr8k_captions(args.flickr_captions_file)
            all_fns = sorted(list(cap_map_all.keys()))

            if args.validate:
                total_groups = args.validate_samples if args.validate_samples > 0 else 1

                # 起始组：文件名优先，其次 validate_start_group
                start_group = args.validate_start_group
                if args.validate_pick_file:
                    try:
                        start_group = all_fns.index(args.validate_pick_file)
                    except ValueError:
                        print(f"[Warn] --validate_pick_file '{args.validate_pick_file}' not found; "
                              f"fallback to --validate_start_group={start_group}")

                print(f"[Info] Validation mode: groups={total_groups} × group_size={args.validate_group_size}, "
                      f"start_group={start_group}, distractors={args.validation_distractors}, "
                      f"warmup={'on' if args.warmup else 'off'}x{args.warmup_iters}")

                images, texts, group_filenames = load_flickr8k_groups_with_names(
                    args.flickr_images_dir,
                    args.flickr_captions_file,
                    total_groups=total_groups,
                    group_size=args.validate_group_size,
                    start_group=start_group,
                )

                image_reprs = normalize_images_bulk(
                    images, args.image_transport, max_workers=args.image_prep_workers
                )

                t0 = time.time()
                embs = enc.encode_v1_multimodal_pairlist_prepared(
                    texts=texts,
                    image_reprs=image_reprs,
                    batch_size=args.batch_size,
                    do_validation=True,
                    validate_group_size=args.validate_group_size,
                    validate_max_samples=total_groups,
                    group_filenames=group_filenames,
                    validation_dump_path=args.validation_dump,
                    global_caption_pool=[(fn, c) for fn, caps in cap_map_all.items() for c in caps],
                    validation_distractors=args.validation_distractors,
                    validate_start_group=0,  # 加载阶段已偏移
                )
            else:
                images, texts = load_flickr8k_pairs(
                    args.flickr_images_dir,
                    args.flickr_captions_file,
                    total=args.num_samples,
                    pick_caption=args.flickr_caption_pick,
                )
                image_reprs = normalize_images_bulk(
                    images, args.image_transport, max_workers=args.image_prep_workers
                )
                t0 = time.time()
                embs = enc.encode_v1_multimodal_pairlist_prepared(
                    texts=texts,
                    image_reprs=image_reprs,
                    batch_size=args.batch_size,
                    do_validation=False,
                )

    elapsed = time.time() - t0
    print("\n==== Summary ====")
    print(f"mode={args.mode}, data_source={args.data_source}, num_samples={args.num_samples}, batch_size={args.batch_size}")
    print(f"shape={tuple(embs.shape)} (dim={embs.shape[1] if embs.numel() else 0})")
    print(f"time(s)={elapsed:.4f}, throughput(samples/s)={args.num_samples / max(elapsed, 1e-9):.2f}")


if __name__ == "__main__":
    main()
