"""
按时间滚动切分数据，避免信息泄漏。
"""


def time_based_split(df, dates: dict) -> dict:
    """
    dates: {"train_end":"2024-12-31", "val_end":"2025-06-30"}
    返回: {"train": df1, "val": df2, "test": df3}
    """
    train_end = dates.get("train_end")
    val_end = dates.get("val_end")
    train = df[df["date"] <= train_end]
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)]
    test = df[df["date"] > val_end]
    return {"train": train, "val": val, "test": test}
