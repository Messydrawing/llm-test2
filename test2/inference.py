"""
Compare teacher (DeepSeek‑R1) and student (Fin‑R1 LoRA) on a single question.
* teacher 调用可选；若缺 API KEY 会返回占位字符串
* student 默认 4‑bit NF4 量化加载，可通过参数替换
* 去除 Prompt 回显，只保留 “### 答案：” 之后的文本
"""

from __future__ import annotations
import argparse, os, re, warnings, json
from pathlib import Path

import torch
from transformers.utils import logging as hf_logging

# ─── 静音 Transformers 的 INFO/WARNING ───────────────────────────────────
hf_logging.set_verbosity_error()
warnings.filterwarnings(
    "ignore", message=r"^Caching is incompatible with gradient checkpointing"
)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from .config import STOCK_CODES, SUMMARY_DAYS
from .data_loader import EastMoneyAPI


# ╭───────────────────────────────────────────────╮
# │                 ⬇ 数据辅助函数                │
# ╰───────────────────────────────────────────────╯
def summarize_stock(question: str, days: int = SUMMARY_DAYS) -> str:
    """从问句里抽取 6 位股票代码并给出近 N 日涨跌幅。"""
    match = re.search(r"\b(\d{6})\b", question)
    code = match.group(1) if match else STOCK_CODES[0]
    api = EastMoneyAPI()
    df = api.get_kline_data(code)
    if df is None or df.empty:
        return ""
    recent = df.tail(days)
    pct = ((recent["close"].iloc[-1] / recent["close"].iloc[0]) - 1) * 100
    return f"【行情提示】股票 {code} 近{days}日涨跌幅：{pct:.2f}%。\n"


# ╭───────────────────────────────────────────────╮
# │           ⬇ 可选 — 远程教师模型调用            │
# ╰───────────────────────────────────────────────╯
ARK_API_KEY = os.getenv("ARK_API_KEY", "...此处填充火山引擎的api...")


def call_teacher(prompt: str) -> dict[str, str]:
    """Call the remote teacher model and return its answer and reasoning."""
    if not ARK_API_KEY:
        return {"content": "[missing ARK_API_KEY]", "reasoning": ""}
    try:
        from volcenginesdkarkruntime import Ark

        client = Ark(api_key=ARK_API_KEY)
        resp = client.chat.completions.create(
            model="deepseek-r1-250528",
            messages=[{"role": "user", "content": prompt}],
        )
        msg = resp.choices[0].message
        content = msg.content.strip()
        reasoning = getattr(msg, "reasoning_content", "").strip()
        return {"content": content, "reasoning": reasoning}
    except Exception as e:  # pragma: no cover - network
        return {"content": f"[teacher model error: {e}]", "reasoning": ""}


# ╭───────────────────────────────────────────────╮
# │               ⬇ 本地学生模型                  │
# ╰───────────────────────────────────────────────╯
def load_student(model_name: str = "Qwen/Qwen1.5-7B"):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_cfg if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    return tokenizer, model.eval()


def call_student(
    tokenizer, model, prompt: str, max_new_tokens: int = 256
) -> str:
    prompt = prompt.rstrip() + "\n\n### 答案："
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    new_tokens = out[0][inputs["input_ids"].size(1) :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ╭───────────────────────────────────────────────╮
# │                  ⬇ 主流程                     │
# ╰───────────────────────────────────────────────╯
def run(question: str, model_path: str | None = None) -> dict[str, object]:
    """拼 Prompt → 调 teacher & student → 返回 dict 结果"""
    prompt = question + "\n" + summarize_stock(question)
    Path("prompt.txt").write_text(prompt + "\n\n### 答案：", encoding="utf-8")

    teacher_answer = call_teacher(prompt)

    tok, mdl = load_student(model_path or "SUFE-AIFLM-Lab/Fin-R1")
    student_answer = call_student(tok, mdl, prompt)

    return {
        "prompt": prompt + "\n\n### 答案：",
        "teacher": teacher_answer,
        "student": student_answer,
    }


# ╭───────────────────────────────────────────────╮
# │                    CLI                        │
# ╰───────────────────────────────────────────────╯
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare teacher & student answer"
    )
    ap.add_argument("question", help="Financial question in Chinese")
    ap.add_argument(
        "--student",
        default="SUFE-AIFLM-Lab/Fin-R1",
        help="Student model or local LoRA dir",
    )
    args = ap.parse_args()

    res = run(args.question, model_path=args.student)
    print("【Prompt】\n" + res["prompt"])
    print("\n【教师模型 DeepSeek‑R1】\n" + res["teacher"]["content"])
    if res["teacher"].get("reasoning"):
        print("\n【教师模型推理】\n" + res["teacher"]["reasoning"])
    print("\n【学生模型】\n" + res["student"])


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
