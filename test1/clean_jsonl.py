#!/usr/bin/env python
# clean_jsonl.py  (v3 — unify label to str)

import argparse, json, pathlib, sys
from collections import Counter
from datasets import Dataset

def normalize_label(lab):
    """return uniform str label or None if invalid"""
    # 原本已是非空字符串
    if isinstance(lab, str) and lab.strip():
        return lab.strip()

    # 字典：必须含有 prediction / analysis / advice 任一键
    if isinstance(lab, dict):
        if any(isinstance(lab.get(k), str) and lab[k].strip()
               for k in ("prediction", "analysis", "advice")):
            return json.dumps(lab, ensure_ascii=False)

    return None  # 其它类型无效

def main(inp: pathlib.Path, out: pathlib.Path):
    stats = Counter()
    good = []

    for raw in inp.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            stats["empty_line"] += 1
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            stats["invalid_json"] += 1
            continue

        prompt = rec.get("prompt", "").strip()
        label  = normalize_label(rec.get("label"))

        if prompt and label:
            good.append(json.dumps({"prompt": prompt, "label": label},
                                   ensure_ascii=False))
            stats["kept"] += 1
        else:
            stats["bad_schema"] += 1

    out.write_text("\n".join(good) + ("\n" if good else ""), "utf-8")
    print(f"[clean_jsonl] wrote {stats['kept']} valid samples → {out}")

    # 再次验证
    Dataset.from_list([json.loads(x) for x in good])
    print("[clean_jsonl] OK   dataset loaded")

    print("\n[clean_jsonl] summary")
    for k, v in stats.items():
        if k != "kept":
            print(f"  {k:12}: {v}")
    print("  kept        :", stats['kept'])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input", type=pathlib.Path)
    p.add_argument("-o", "--output", type=pathlib.Path,
                   help="cleaned file path (default cleaned_<input>.jsonl)")
    args = p.parse_args()

    inp = args.input.resolve()
    if not inp.is_file():
        sys.exit(f"Input not found: {inp}")

    out = args.output or inp.with_name(f"cleaned_{inp.name}")
    main(inp, out)
