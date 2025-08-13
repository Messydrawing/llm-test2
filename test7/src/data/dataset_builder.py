"""
把 raw K 线 -> processed prompt 样本，并落地为 {train,val,test}.jsonl
"""
import json
from .schema import PromptItem


def format_prompt(
    template: str, *, stock_code: str, kline_summary, change: float
) -> str:
    """将结构化数据嵌入模板，生成最终提示串。"""
    summary_json = json.dumps(kline_summary, ensure_ascii=False)
    return template.format(
        stock_code=stock_code, summary=summary_json, change=round(change, 2)
    )


def build_prompts_from_kline(kline_rows, template: str) -> PromptItem:
    """
    使用固定模板构建最终 prompt:
    "股票 {stock_code} 近30日K线数据: {summary}\n涨跌幅: {change}%。
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
    kline_summary = window_df.to_dict(orient="records")
    stock_code = kline_rows[0].get("stock_code", "")
    prompt = format_prompt(
        template,
        stock_code=stock_code,
        kline_summary=kline_summary,
        change=change,
    )
    return PromptItem(
        stock_code=stock_code,
        kline_summary=kline_summary,
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
