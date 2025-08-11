"""
把 raw K 线 -> processed prompt 样本，并落地为 {train,val,test}.jsonl
"""
from .schema import PromptItem


def build_prompts_from_kline(kline_rows, template: str) -> PromptItem:
    """
    使用固定模板构建最终 prompt:
    "股票 {stock_code} 近30日K线数据: {summary}\n涨跌幅: {change}%。
     请预测后市走势，给出简短分析和操作建议，并以 JSON 格式回复，
     包括 'prediction', 'analysis', 'advice' 三个字段。"
    返回 PromptItem
    """
    raise NotImplementedError


def save_jsonl(objs, path: str):
    """将对象列表以 JSONL 格式写入磁盘。"""
    raise NotImplementedError


def time_roll_split(df, cutoff_dates) -> dict:
    """
    按给定日期分三段: 训练/验证/测试，返回路径或 DataFrame 映射。
    """
    raise NotImplementedError
