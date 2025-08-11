"""
把 raw K 线 -> processed prompt 样本，并落地为 {train,val,test}.jsonl
"""
import json
from .schema import PromptItem
from .summarize_kline import make_summary


def build_prompts_from_kline(kline_rows, template: str) -> PromptItem:
    """
    使用固定模板构建最终 prompt:
    "股票 {stock_code} 近30日K线数据: {summary}\n涨跌幅: {change}%。
     请预测后市走势，给出简短分析和操作建议，并以 JSON 格式回复，
     包括 'prediction', 'analysis', 'advice' 三个字段。"
    返回 PromptItem
    """
    summary, change, _ = make_summary(kline_rows)
    stock_code = kline_rows[0].get("stock_code", "") if kline_rows else ""
    prompt = template.format(
        stock_code=stock_code, summary=summary, change=round(change, 2)
    )
    return PromptItem(
        stock_code=stock_code, summary=summary, change=change, prompt=prompt
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
