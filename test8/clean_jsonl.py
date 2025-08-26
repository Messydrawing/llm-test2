#!/usr/bin/env python
"""Remove records without valid ``prompt``/``label`` fields from a JSONL file."""

import argparse
import json
import pathlib
from collections import Counter


def normalize_label(lab):
    if isinstance(lab, str) and lab.strip():
        return lab.strip()
    if isinstance(lab, dict):
        has_field = any(
            isinstance(lab.get(k), str) and lab[k].strip()
            for k in ("prediction", "analysis", "advice")
        )
        if has_field:
            return lab
    return None


def main(inp: pathlib.Path, out: pathlib.Path) -> None:
    stats = Counter()
    good = []
    for raw in inp.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            stats["empty"] += 1
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            stats["bad_json"] += 1
            continue
        prompt = rec.get("prompt", "").strip()
        label = normalize_label(rec.get("label"))
        if prompt and label is not None:
            good.append(json.dumps({"prompt": prompt, "label": label}, ensure_ascii=False))
            stats["kept"] += 1
        else:
            stats["bad_schema"] += 1
    out.write_text("\n".join(good) + ("\n" if good else ""), encoding="utf-8")
    print(f"[clean_jsonl] wrote {stats['kept']} samples → {out}")
    if stats["bad_json"] or stats["bad_schema"]:
        print("[clean_jsonl] discarded:")
        for k in ("empty", "bad_json", "bad_schema"):
            if stats[k]:
                print(f"  {k}: {stats[k]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input", type=pathlib.Path)
    p.add_argument("-o", "--output", type=pathlib.Path, help="cleaned file path")
    args = p.parse_args()
    inp = args.input.resolve()
    out = args.output or inp.with_name(f"cleaned_{inp.name}")
    main(inp, out)
