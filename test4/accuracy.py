from __future__ import annotations

import argparse
import datetime
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from .config import STOCK_CODES
from .data_loader import EastMoneyAPI


def sample_windows(num_samples: int = 100) -> List[Dict[str, Any]]:
    api = EastMoneyAPI()
    today = datetime.date.today()
    cutoff = (today - datetime.timedelta(days=7)).strftime("%Y-%m-%d")

    per_label = {"涨": [], "跌": []}
    required = num_samples // 2

    codes = list(STOCK_CODES)
    random.shuffle(codes)
    for code in codes:
        if all(len(v) >= required for v in per_label.values()):
            break
        df = api.get_kline_data(code, num=1000)
        if df is None:
            continue
        n = len(df)
        if n < 37:
            continue
        for start in range(n - 37 + 1):
            if all(len(v) >= required for v in per_label.values()):
                break
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
            if len(per_label[label]) >= required:
                continue
            per_label[label].append(
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
    samples = per_label["涨"][:required] + per_label["跌"][:required]
    random.shuffle(samples)
    return samples


def build_prompt(sample: Dict[str, Any]) -> str:
    data_str = json.dumps(sample["data"], ensure_ascii=False)
    t0 = sample["data"][0]["close"]
    t1 = sample["data"][-1]["close"]
    change_pct = (t1 - t0) / t0 * 100
    return (
        f"股票 {sample['code']} 近30日K线数据: {data_str}\n"
        f"涨跌幅: {change_pct:.2f}%。请预测后市走势，给出简短分析和操作建议，并以 JSON 格式回复，包括 'prediction', 'analysis', 'advice' 三个字段。"
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


def extract_prediction(text: str) -> str:
    try:
        data = json.loads(text)
        return str(data.get("prediction", ""))
    except Exception:
        return text


def resolve_direction(text: str) -> Optional[str]:
    for ch in text:
        if ch in ("涨", "上", "增"):
            return "涨"
        if ch in ("跌", "下", "降"):
            return "跌"
    return None


def evaluate(
    tokenizer,
    model,
    device,
    samples: List[Dict[str, Any]],
    max_new_tokens: int = 512,
) -> float:
    if not samples:
        return 0.0
    correct = 0.0
    cm = {"涨": {"涨": 0, "跌": 0, None: 0}, "跌": {"涨": 0, "跌": 0, None: 0}}

    for s in samples:
        prompt = build_prompt(s)
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": "你是一个只回答JSON的助手。"},
                {"role": "user", "content": prompt},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        # Determine how many tokens we can still generate without exceeding
        # the model's context window. This prevents truncation while keeping
        # generation within the model limits.
        ctx_len = inputs["input_ids"].shape[1]
        max_ctx = getattr(model.config, "max_position_embeddings", ctx_len + max_new_tokens)
        gen_tokens = min(max_new_tokens, max_ctx - ctx_len)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=gen_tokens,
                max_length=min(max_ctx, ctx_len + gen_tokens),
                do_sample=False,
                temperature=0,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer_raw = tokenizer.decode(
            out[0][ctx_len:], skip_special_tokens=True
        )
        answer = answer_raw.strip()

        pred_text = extract_prediction(answer)
        pred = resolve_direction(pred_text)

        if pred == s["label"]:
            correct += 1
            result = "预测正确"
        elif pred is None:
            correct += 0.5
            result = "无法判断，记0.5分"
        else:
            result = "预测错误"

        cm[s["label"]][pred] += 1
        direction = "涨" if s["change"] >= 0 else "跌"
        print(
            f"code: {s['code']}, t0: {s['t0_date']}/{s['t0_close']:.2f}, "
            f"t1: {s['t1_date']}/{s['t1_close']:.2f}, t2: {s['t2_date']}/{s['t2_close']:.2f}"
        )
        print(f"raw answer: {answer_raw!r}")
        print(
            f"change: {s['change']:.2f} ({direction}), pred_text: {pred_text}, "
            f"pred: {pred}, {result}"
        )

    print("\nConfusion matrix (true rows, pred cols):")
    print("\t涨\t跌\tNone")
    for true in ("涨", "跌"):
        row = cm[true]
        print(f"{true}\t{row['涨']}\t{row['跌']}\t{row[None]}")

    return correct / len(samples)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate student LoRA accuracy")
    ap.add_argument("--base", required=True, help="Base model directory")
    ap.add_argument("--student", required=True, help="LoRA directory")
    ap.add_argument("--out", help="Optional JSONL output path")
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate for each sample",
    )
    args = ap.parse_args()

    random.seed(0)
    samples = sample_windows()
    if args.out:
        path = Path(args.out)
        with path.open("w", encoding="utf-8") as f:
            for rec in samples:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tokenizer, model, device = load_model(args.base, args.student)
    acc = evaluate(tokenizer, model, device, samples, max_new_tokens=args.max_new_tokens)
    print(f"Accuracy: {acc:.2%}")
    if args.out:
        print(f"Saved samples to {args.out}")


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
