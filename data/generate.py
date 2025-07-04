import numpy as np
from typing import List, Tuple


def generate_stock_series(
    num_series: int, seq_len: int, seed: int | None = None
) -> List[np.ndarray]:
    """Generate synthetic stock price series."""
    rng = np.random.default_rng(seed)
    series = []
    for _ in range(num_series):
        # simple geometric Brownian motion
        prices = [1.0]
        for _ in range(seq_len - 1):
            change = rng.normal(scale=0.01)
            prices.append(prices[-1] * (1 + change))
        series.append(np.array(prices, dtype=np.float32))
    return series


def make_dataset(
    num_train: int, num_val: int, seq_len: int, seed: int | None = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Create train and validation datasets."""
    train = generate_stock_series(num_train, seq_len, seed)
    val = generate_stock_series(
        num_val, seq_len, seed + 1 if seed is not None else None
    )
    return train, val
