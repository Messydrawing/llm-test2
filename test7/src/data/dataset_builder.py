"""把 raw K 线 -> processed prompt 样本，并落地为 {train,val,test}.jsonl."""

import json
from typing import Any

from .schema import PromptItem


def format_prompt(
    template: str, *, stock_code: str, kline_json, change: float
) -> str:
    """将结构化数据嵌入模板，生成最终提示串。"""
    kline_json_str = json.dumps(kline_json, ensure_ascii=False)
    return template.format(
        stock_code=stock_code, kline_json=kline_json_str, change=round(change, 2)
    )


def _trim_sample_tokens(sample: dict[str, Any], tokenizer, max_tokens: int) -> None:
    """Trim ``kline_json`` so ``format_prompt(sample)`` fits ``max_tokens``."""
    if tokenizer is None:
        return
    while len(sample["kline_json"]) > 1:
        text = format_prompt(
            sample["template"],
            stock_code=sample["stock_code"],
            kline_json=sample["kline_json"],
            change=sample["change"],
        )
        if (
            len(tokenizer(text, add_special_tokens=False)["input_ids"])
            <= max_tokens
        ):
            break
        sample["kline_json"].pop(0)


def build_prompts_from_kline(
    kline_rows,
    template: str,
    *,
    tokenizer=None,
    max_tokens: int = 1024,
) -> PromptItem:
    """
    使用固定模板构建最终 prompt:
    "股票 {stock_code} 近30日K线数据: {kline_json}\n涨跌幅: {change}%。
     请预测后市走势，给出简短分析和操作建议，并以 JSON 格式回复，
     包括 'prediction', 'analysis', 'advice' 三个字段。"
    返回 PromptItem
    """
    if not kline_rows:
        raise ValueError("kline_rows is empty")
    import pandas as pd

    window_df = pd.DataFrame(kline_rows)[
        [
            "date",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "MA5",
            "MA10",
            "RSI14",
            "MACD",
        ]
    ]
    change = ((window_df["close"].iloc[-1] / window_df["close"].iloc[0]) - 1) * 100
    kline_json = window_df.to_dict(orient="records")
    stock_code = kline_rows[0].get("stock_code", "")
    sample = {
        "stock_code": stock_code,
        "kline_json": kline_json,
        "change": change,
        "template": template,
    }
    _trim_sample_tokens(sample, tokenizer, max_tokens)
    prompt = format_prompt(
        template,
        stock_code=stock_code,
        kline_json=sample["kline_json"],
        change=change,
    )
    return PromptItem(
        stock_code=stock_code,
        kline_json=sample["kline_json"],
        change=change,
        prompt=prompt,
    )


def save_jsonl(objs, path: str):
    """将对象列表以 JSONL 格式写入磁盘。"""
    with open(path, "w", encoding="utf-8") as f:
        for obj in objs:
            if hasattr(obj, "model_dump"):
                obj = obj.model_dump()
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def time_roll_split(df, cutoff_dates) -> dict:
    """
    按给定日期分三段: 训练/验证/测试，返回路径或 DataFrame 映射。
    """
    train_end = cutoff_dates.get("train_end")
    val_end = cutoff_dates.get("val_end")
    train = df[df["date"] <= train_end]
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)]
    test = df[df["date"] > val_end]
    return {"train": train, "val": val, "test": test}
