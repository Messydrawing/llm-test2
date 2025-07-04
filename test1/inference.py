from __future__ import annotations

import argparse
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import STOCK_CODES, SUMMARY_DAYS
from .data_loader import EastMoneyAPI


def summarize_stock(question: str, days: int = SUMMARY_DAYS) -> str:
    match = re.search(r"\b(\d{6})\b", question)
    code = match.group(1) if match else STOCK_CODES[0]
    api = EastMoneyAPI()
    df = api.get_kline_data(code)
    if df is None or df.empty:
        return ""
    recent = df.tail(days)
    pct = ((recent["close"].iloc[-1] / recent["close"].iloc[0]) - 1) * 100
    return f"{code} has changed {pct:.2f}% in the last {days} days."


def call_teacher(prompt: str) -> str:
    api_key = "ff6a5d1b-beef-4b53-aa49-7015da1693a1"
    if not api_key:
        return "[missing ARK_API_KEY]"
    try:
        from volcenginesdkarkruntime import Ark

        client = Ark(api_key=api_key)
        resp = client.chat.completions.create(
            model="deepseek-r1-250528",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    except Exception as e:  # pragma: no cover - network
        return f"[teacher model error: {e}]"


def load_student(model_name: str = "SUFE-AIFLM-Lab/Fin-R1"):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=False, trust_remote_code=True
    )
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    else:
        print("[Warning] GPU not available, using CPU for student model.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            trust_remote_code=True,
        )
    return tokenizer, model


def call_student(tokenizer, model, prompt: str) -> str:
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model.generate(**enc, max_new_tokens=512, temperature=0.7, top_p=0.8)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def run(question: str) -> dict[str, str]:
    info = summarize_stock(question)
    prompt = question if not info else f"{question}\n{info}"
    with open("prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)
    teacher = call_teacher(prompt)
    tokenizer, model = load_student()
    student = call_student(tokenizer, model, prompt)
    return {"question": question, "teacher": teacher, "student": student}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare teacher and student")
    parser.add_argument("question", help="Financial question")
    args = parser.parse_args()
    result = run(args.question)
    print("教师模型回答:")
    print(result["teacher"])
    print("\n学生模型回答:")
    print(result["student"])


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
