import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from test4.dataset_builder import build_dataset, format_prompt
from test4.config import STOCK_CODES


def _label_completion(change: float) -> dict:
    """Create a simple completion based on price change."""
    if change > 3:
        return {
            "prediction": "up",
            "analysis": f"30\u65e5\u5185\u4e0a\u6da8{change:.2f}%, \u77e9\u73af\u4e0a\u6da8\u8d8b\u52bf\u660e\u663e.",
            "advice": "\u8003\u8651\u8fdb\u573a\u62ff\u4ed3",
        }
    if change < -3:
        return {
            "prediction": "down",
            "analysis": f"30\u65e5\u5185\u4e0b\u8dcc{abs(change):.2f}%, \u60ac\u5d16\u76d8\u7a7a\u529b\u65e0\u76d6.",
            "advice": "\u8003\u8651\u51fa\u573a\u89e3\u4ed3",
        }
    return {
        "prediction": "stable",
        "analysis": f"30\u65e5\u5185\u6da8\u8dcc{change:.2f}%, \u5927\u81f4\u6b63\u5e38\u6da8\u8dcc\u5e45\u5ea6.",
        "advice": "\u4fdd\u6301\u89c2\u671b",
    }


def _convert_samples(samples: Sequence[dict]) -> list[dict]:
    data = []
    for s in samples:
        prompt = format_prompt(s)
        completion = json.dumps(_label_completion(s["change"]), ensure_ascii=False)
        data.append({"prompt": prompt, "completion": completion})
    return data


def write_jsonl(records: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(output_dir: str = "test9/data") -> None:
    train_samples, val_samples = build_dataset(
        STOCK_CODES, days=120, window=30, windows_per_stock=1, val_ratio=0.2, seed=42
    )
    mid = len(val_samples) // 2
    val_set = val_samples[:mid]
    test_set = val_samples[mid:]

    train_records = _convert_samples(train_samples)
    val_records = _convert_samples(val_set)
    test_records = _convert_samples(test_set)

    out_dir = Path(output_dir)
    write_jsonl(train_records, out_dir / "train.jsonl")
    write_jsonl(val_records, out_dir / "val.jsonl")
    write_jsonl(test_records, out_dir / "test.jsonl")


if __name__ == "__main__":
    main()
