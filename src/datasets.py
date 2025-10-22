import os, re, glob
from typing import List, Tuple
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset

def load_pairs(dataset_key: str, split: str, max_samples: int = -1):
    """
    支持：
      - 远端：C-MTEB/LCQMC、clue/afqmc、paws-x:zh
      - 本地：本地 Hub 仓库目录（优先 load_dataset(dir)）、save_to_disk 目录（load_from_disk）、
              以及递归扫描 csv/tsv/json/jsonl/parquet（按 split 关键字或回退）
    需要的列：sentence1/sentence2 + label|score （自动别名识别）
    """
    key = dataset_key
    key_l = dataset_key.lower()

    # ---- 远端内置数据集 ----
    if key_l == "lcqmc":
        ds = load_dataset("C-MTEB/LCQMC", split=split)
        if max_samples > 0: ds = ds.select(range(min(max_samples, len(ds))))
        return [r["sentence1"] for r in ds], [r["sentence2"] for r in ds], [int(r["score"]) for r in ds]

    if key_l == "afqmc":
        ds = load_dataset("clue", name="afqmc", split=split)
        if max_samples > 0: ds = ds.select(range(min(max_samples, len(ds))))
        return [r["sentence1"] for r in ds], [r["sentence2"] for r in ds], [int(r["label"]) for r in ds]

    if key_l in ("pawsx-zh", "pawsx", "paws-x-zh"):
        ds = load_dataset("paws-x", name="zh", split=split)
        if max_samples > 0: ds = ds.select(range(min(max_samples, len(ds))))
        return [r["sentence1"] for r in ds], [r["sentence2"] for r in ds], [int(r["label"]) for r in ds]

    # ---- 本地路径 ----
    if not os.path.exists(key):
        raise ValueError(f"Unknown dataset or path not found: {dataset_key}")

    def _pick_split(ds_like, want_split: str):
        """根据 want_split 在 DatasetDict 中优先寻找 ['validation','valid','dev','test','train'] 的别名"""
        if isinstance(ds_like, Dataset):
            return ds_like
        if not isinstance(ds_like, DatasetDict):
            raise ValueError(f"Unexpected dataset type: {type(ds_like)}")

        want = want_split.lower()
        candidates = [want]
        if want == "validation":
            candidates += ["valid", "dev"]
        candidates += ["validation", "valid", "dev", "test", "train"]
        for c in candidates:
            if c in ds_like:
                return ds_like[c]
        return next(iter(ds_like.values()))

    def _detect_cols(names):
        names_l = [n.lower() for n in names]
        pair_candidates = [
            ("sentence1", "sentence2"),
            ("text1", "text2"),
            ("question1", "question2"),
            ("q", "p"),
            ("s1", "s2"),
            ("query", "passage"),
        ]
        label_candidates = ["label", "score", "target", "y", "similar"]

        s1 = s2 = lbl = None
        for a, b in pair_candidates:
            if a in names_l and b in names_l:
                s1 = names[names_l.index(a)]
                s2 = names[names_l.index(b)]
                break
        for c in label_candidates:
            if c in names_l:
                lbl = names[names_l.index(c)]
                break
        if s1 is None or s2 is None or lbl is None:
            raise ValueError(f"Cannot find pair/label columns in {names}")
        return s1, s2, lbl

    def _finalize(ds):
        if max_samples > 0:
            ds = ds.select(range(min(max_samples, len(ds))))
        s1_col, s2_col, y_col = _detect_cols(ds.column_names)
        s1 = [r[s1_col] for r in ds]
        s2 = [r[s2_col] for r in ds]
        ys = []
        for v in ds[y_col]:
            if isinstance(v, bool):
                ys.append(int(v))
            else:
                try:
                    fv = float(v)
                    ys.append(int(fv >= 0.5))
                except Exception:
                    vv = str(v).strip().lower()
                    if vv in ("1", "true", "yes", "y"):
                        ys.append(1)
                    elif vv in ("0", "false", "no", "n"):
                        ys.append(0)
                    else:
                        raise ValueError(f"Unrecognized label value: {v!r}")
        return s1, s2, ys

    if os.path.isdir(key):
        # 目录 -> 先尝试作为本地 Hub 仓库
        try:
            ds = load_dataset(key, split=split)
            return _finalize(ds)
        except Exception:
            pass

        # 再尝试 save_to_disk
        try:
            ds_or_dict = load_from_disk(key)
            ds = _pick_split(ds_or_dict, split)
            return _finalize(ds)
        except Exception:
            pass

        # 递归搜文件
        exts = ("csv","tsv","json","jsonl","parquet")
        split_alias = [split.lower()]
        if split.lower() == "validation":
            split_alias += ["valid", "dev"]
        split_alias += ["validation", "valid", "dev", "test", "train"]
        pattern = re.compile(r"(" + "|".join(map(re.escape, split_alias)) + r")", flags=re.IGNORECASE)

        cand_files = []
        for ext in exts:
            cand_files += glob.glob(os.path.join(key, "**", f"*.{ext}"), recursive=True)

        matched = [f for f in cand_files if pattern.search(os.path.basename(f))]
        if not matched and cand_files:
            matched = cand_files

        if not matched:
            raise ValueError(f"No split file found under dir {key}")

        key = matched[0]

    # 单文件
    ext = os.path.splitext(key)[1].lower().lstrip(".")
    reader_map = {"csv": "csv", "tsv": "csv", "json": "json", "jsonl": "json", "parquet": "parquet"}
    if ext not in reader_map:
        raise ValueError(f"Unsupported local file type: .{ext} (support: csv/tsv/json/jsonl/parquet)")

    data_files = {split: key}
    if ext == "tsv":
        ds = load_dataset("csv", data_files=data_files, delimiter="\t")[split]
    else:
        ds = load_dataset(reader_map[ext], data_files=data_files)[split]
    return _finalize(ds)

# ========= Yahoo Answers (JSONL) =========
def load_yahoo_answers_jsonl(path: str, mode: str = "q", max_records: int = -1):
    """
    读取 yahoo_answers_title_answer.jsonl（或同类 JSONL）。
    mode:
      'q'   -> 只取 question
      'q+a' -> question 与 answer 各生成一条文本（单独编码）

    支持多种行格式：
      1) dict:
         - question: 'question', 'title', 'query'
         - answer:   'answer', 'best_answer', 'response'
         - answers:  列表（元素可为 str 或 dict，dict 优先取 'text'/'answer'/'content'）
      2) list/tuple:
         - ["question", "answer", ...]（字符串列表）
         - [{"text": "question"}, {"text": "answer"}]（对象列表）

    max_records: 仅处理前 N 条 JSON 行（-1 代表不限制，全部处理）

    返回: List[str] 文本列表（无标签）
    """
    mode = (mode or "q").lower()
    assert mode in ("q", "q+a"), "mode must be 'q' or 'q+a'"

    texts = []
    import json

    def pick_dict_field(d: dict, keys):
        for k in keys:
            if k in d and d[k] is not None:
                v = d[k]
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return None

    def first_non_empty_from_answers(ans):
        """
        ans 可能是：
          - str
          - dict（取 'text'/'answer'/'content'/'best_answer'/'response'）
          - list[ str | dict ]
        返回第一条非空字符串；找不到则 None
        """
        if isinstance(ans, str) and ans.strip():
            return ans.strip()

        if isinstance(ans, dict):
            return pick_dict_field(ans, ["text", "answer", "content", "best_answer", "response"])

        if isinstance(ans, (list, tuple)):
            for x in ans:
                if isinstance(x, str) and x.strip():
                    return x.strip()
            for x in ans:
                if isinstance(x, dict):
                    v = pick_dict_field(x, ["text", "answer", "content", "best_answer", "response"])
                    if v:
                        return v
        return None

    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if max_records > 0 and count >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            q_text = None
            a_text = None

            if isinstance(obj, dict):
                q_text = pick_dict_field(obj, ["question", "title", "query", "q", "content", "text"])
                a_text = pick_dict_field(obj, ["answer", "best_answer", "response", "a", "accepted_answer"])
                if not a_text and "answers" in obj:
                    a_text = first_non_empty_from_answers(obj["answers"])

            elif isinstance(obj, (list, tuple)) and len(obj) > 0:
                first_item = obj[0]
                second_item = obj[1] if len(obj) > 1 else None

                if isinstance(first_item, str) and first_item.strip():
                    q_text = first_item.strip()
                elif isinstance(first_item, dict):
                    q_text = pick_dict_field(first_item, ["question", "title", "query", "q", "content", "text"])

                if mode == "q+a" and second_item is not None:
                    a_text = first_non_empty_from_answers(second_item)

            # 一条 JSON 记录处理完计数 +1（不看 q/a 实际是否写入 texts）
            count += 1

            if q_text:
                texts.append(q_text)
            if mode == "q+a" and a_text:
                texts.append(a_text)

    return texts
