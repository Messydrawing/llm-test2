from __future__ import annotations

import argparse
import datetime
import json
import random
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from .config import STOCK_CODES
from .data_loader import EastMoneyAPI


def sample_windows(num_samples: int = 100) -> List[Dict[str, Any]]:
    api = EastMoneyAPI()
    today = datetime.date.today()
    cutoff = (today - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    samples: List[Dict[str, Any]] = []
    for code in STOCK_CODES:
        df = api.get_kline_data(code, num=1000)
        if df is None:
            continue
        n = len(df)
        if n < 37:
            continue
        for start in range(n - 37 + 1):
            end = start + 29
            t1_date = df["date"].iloc[end]
            if t1_date > cutoff:
                continue
            win = df.iloc[start : start + 30].reset_index(drop=True)
            t0_close = win["close"].iloc[0]
            t1_close = win["close"].iloc[-1]
            t2_close = df["close"].iloc[end + 7]
            t0_date = win["date"].iloc[0]
            t2_date = df["date"].iloc[end + 7]
            change = t1_close - t0_close
            label = "跌" if t1_close > t2_close else "涨"
            samples.append(
                {
                    "code": code,
                    "data": win.to_dict(orient="records"),
                    "label": label,
                    "t0_date": t0_date,
                    "t0_close": t0_close,
                    "t1_date": t1_date,
                    "t1_close": t1_close,
                    "t2_date": t2_date,
                    "t2_close": t2_close,
                    "change": change,
                }
            )
    random.shuffle(samples)
    return samples[:num_samples]


def build_prompt(sample: Dict[str, Any]) -> str:
    data_str = json.dumps(sample["data"], ensure_ascii=False)
    t0 = sample["data"][0]["close"]
    t1 = sample["data"][-1]["close"]
    change = t1 - t0
    return (
        f"股票代码“{sample['code']}”，数据：{data_str}，涨跌幅：{change:.2f}，"
        "请你基于以上数据，判断股票未来7天的涨跌趋势，输出“涨”或“跌”。"
    )


def load_model(base: str, student: str):
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, trust_remote_code=True, torch_dtype=torch_dtype
    )
    tokenizer.pad_token = tokenizer.eos_token
    if hasattr(PeftModel, "from_pretrained"):
        try:
            model = PeftModel.from_pretrained(model, student)
        except Exception as e:  # pragma: no cover - optional lora
            print(f"[warning] failed to load LoRA from {student}: {e}")
    else:  # pragma: no cover - stub fallback
        if student != base:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    student, trust_remote_code=True, torch_dtype=torch_dtype
                )
            except Exception as e:
                print(f"[warning] failed to load student model from {student}: {e}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device


def evaluate(tokenizer, model, device, samples: List[Dict[str, Any]]) -> float:
    if not samples:
        return 0.0
    correct = 0
    for s in samples:
        prompt = build_prompt(s)
        max_len = max(8, tokenizer.model_max_length - 5)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        ).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=5)
        answer = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        if "涨" in answer and "跌" not in answer:
            pred = "涨"
        elif "跌" in answer and "涨" not in answer:
            pred = "跌"
        else:
            pred = None

        if pred == s["label"]:
            correct += 1
            result = "预测正确"
        else:
            result = "预测错误"

        direction = "涨" if s["change"] >= 0 else "跌"
        print(
            f"code: {s['code']}, t0: {s['t0_date']}/{s['t0_close']:.2f}, "
            f"t1: {s['t1_date']}/{s['t1_close']:.2f}, t2: {s['t2_date']}/{s['t2_close']:.2f}"
        )
        print(
            f"change: {s['change']:.2f} ({direction}), answer: {answer}, "
            f"pred: {pred}, {result}"
        )
    return correct / len(samples)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate student LoRA accuracy")
    ap.add_argument("--base", required=True, help="Base model directory")
    ap.add_argument("--student", required=True, help="LoRA directory")
    ap.add_argument("--out", help="Optional JSONL output path")
    args = ap.parse_args()

    random.seed(0)
    samples = sample_windows()
    if args.out:
        path = Path(args.out)
        with path.open("w", encoding="utf-8") as f:
            for rec in samples:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tokenizer, model, device = load_model(args.base, args.student)
    acc = evaluate(tokenizer, model, device, samples)
    print(f"Accuracy: {acc:.2%}")
    if args.out:
        print(f"Saved samples to {args.out}")


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
