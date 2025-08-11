"""
按时间滚动切分数据，避免信息泄漏。
"""


def time_based_split(df, dates: dict) -> dict:
    """
    dates: {"train_end":"2024-12-31", "val_end":"2025-06-30"}
    返回: {"train": df1, "val": df2, "test": df3}
    """
    raise NotImplementedError
