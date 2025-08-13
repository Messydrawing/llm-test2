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


def build_dataset(
    stock_codes,
    *,
    days: int = 180,
    window: int = 30,
    windows_per_stock: int | None = None,
    val_ratio: float = 0.2,
    balance: bool = True,
    seed: int | None = None,
    template: str = (
        "股票 {stock_code} 近30日K线数据: {kline_json}\n"
        "涨跌幅: {change}%。请预测后市走势，给出简短分析和操作建议，并以 JSON 格式回复，包括 'prediction', 'analysis', 'advice' 三个字段。"
    ),
    tokenizer=None,
    max_tokens: int = 1024,
):
    """构建训练/验证集 PromptItem 列表。

    通过 :class:`EastMoneyAPI` 获取每只股票最近 ``days`` 天的K线数据，
    按 ``window`` 进行滑窗切片，并根据涨跌幅划分 up/down/stable 三类。
    当 ``balance=True`` 时，会对三类样本进行下采样以保持平衡。
    返回训练集与验证集两个 :class:`PromptItem` 列表。
    """

    import random
    from typing import Sequence

    from .eastmoney_client import EastMoneyAPI

    try:  # optional heavy deps
        import pandas as pd  # noqa: F401
        import numpy as np  # noqa: F401
    except Exception as e:  # pragma: no cover - runtime guard
        raise ImportError("pandas and numpy are required for dataset building") from e

    codes: Sequence[str] = list(stock_codes)
    rng = random.Random(seed)
    api = EastMoneyAPI()

    up_items: list[PromptItem] = []
    down_items: list[PromptItem] = []
    stable_items: list[PromptItem] = []

    for code in codes:
        df = api.get_kline_data(code, num=days)
        if df is None or df.empty:
            continue

        df["pct_chg"] = df["close"].pct_change() * 100
        df["pct_chg"] = df["pct_chg"].fillna(0)
        df["MA5"] = df["close"].rolling(5, min_periods=5).mean()
        df["MA10"] = df["close"].rolling(10, min_periods=10).mean()
        diff = df["close"].diff()
        gains = diff.clip(lower=0)
        losses = -diff.clip(upper=0)
        avg_gain = gains.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1 / 14, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = np.where(
            avg_loss.to_numpy() == 0,
            np.where(avg_gain.to_numpy() == 0, 50, 100),
            100 - 100 / (1 + rs.to_numpy()),
        )
        df["RSI14"] = rsi
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df[["MA5", "MA10", "RSI14", "MACD"]] = df[
            ["MA5", "MA10", "RSI14", "MACD"]
        ].round(2)

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])

        n = len(df)
        if n < window:
            continue

        valid_indices = [
            i
            for i in range(n - window + 1)
            if not (
                df["volume"].iloc[i : i + window].eq(0).any()
                or df["pct_chg"].iloc[i : i + window].abs().gt(20).any()
            )
        ]
        if windows_per_stock is not None and len(valid_indices) > windows_per_stock:
            valid_indices = rng.sample(valid_indices, windows_per_stock)

        for i in valid_indices:
            win = df.iloc[i : i + window][
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
            ].reset_index(drop=True)
            win["date"] = pd.to_datetime(win["date"]).dt.strftime("%Y-%m-%d")
            win["stock_code"] = code

            change_pct = ((win["close"].iloc[-1] / win["close"].iloc[0]) - 1) * 100
            try:
                item = build_prompts_from_kline(
                    win.to_dict(orient="records"),
                    template,
                    tokenizer=tokenizer,
                    max_tokens=max_tokens,
                )
            except ValueError:
                continue
            if change_pct > 3:
                up_items.append(item)
            elif change_pct < -3:
                down_items.append(item)
            else:
                stable_items.append(item)

    if balance and up_items and down_items and stable_items:
        min_count = min(len(up_items), len(down_items), len(stable_items))
        rng.shuffle(up_items)
        rng.shuffle(down_items)
        rng.shuffle(stable_items)
        up_items = up_items[:min_count]
        down_items = down_items[:min_count]
        stable_items = stable_items[:min_count]

    samples = up_items + down_items + stable_items
    rng.shuffle(samples)
    split = int(len(samples) * (1 - val_ratio))
    return samples[:split], samples[split:]
