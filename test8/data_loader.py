import datetime
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd
import requests


class EastMoneyAPI:
    """Wrapper around the EastMoney K-line API with optional local caching.

    Parameters
    ----------
    cache_dir: str or Path, optional
        Directory to store cached csv files. Defaults to ``test8/cache``.
    use_cache: bool
        Whether to read/write cache files. Defaults to ``True``.
    """

    KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

    def __init__(self, cache_dir: str | Path | None = None, use_cache: bool = True) -> None:
        self.session = requests.Session()
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir or Path(__file__).with_name("cache"))
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _secid(self, code: str) -> str:
        return f"{'1' if code.startswith('6') else '0'}.{code}"

    def _cache_file(self, code: str, klt: int, num: int) -> Path:
        return self.cache_dir / f"{code}_{klt}_{num}.csv"

    # ------------------------------------------------------------------
    def get_kline_data(
        self, stock_code: str, *, klt: int = 101, num: int = 1000, refresh: bool = False
    ) -> pd.DataFrame | None:
        """Fetch kline data for ``stock_code``.

        Data are cached locally so subsequent calls with the same arguments can
        run offline. When ``refresh`` is True a new network request is forced.
        """

        cache_path = self._cache_file(stock_code, klt, num)
        if self.use_cache and cache_path.exists() and not refresh:
            try:
                return pd.read_csv(cache_path)
            except Exception:
                pass  # Fall back to network fetch

        params = {
            "secid": self._secid(stock_code),
            "klt": klt,
            "fqt": 0,
            "lmt": num,
            "end": "20500000",
            "beg": "0",
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
        }
        try:
            r = self.session.get(self.KLINE_URL, params=params, timeout=8)
            r.raise_for_status()
            js = r.json()
            klines = js.get("data", {}).get("klines", [])
            records = []
            for line in klines:
                d = line.split(",")
                records.append(
                    {
                        "date": d[0],
                        "open": float(d[1]),
                        "close": float(d[2]),
                        "high": float(d[3]),
                        "low": float(d[4]),
                        "volume": float(d[5]),
                    }
                )
            df = pd.DataFrame(records)
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            if self.use_cache:
                df.to_csv(cache_path, index=False)
            return df
        except Exception as e:  # pragma: no cover - network issues
            if self.use_cache and cache_path.exists():
                try:
                    return pd.read_csv(cache_path)
                except Exception:
                    pass
            print(f"[EastMoneyAPI] {stock_code} fetch failed: {e}")
            return None


# ----------------------------------------------------------------------
def get_recent_data(
    stock_codes: Sequence[str],
    *,
    days: int = 90,
    klt: int = 101,
    api: EastMoneyAPI | None = None,
) -> Dict[str, pd.DataFrame]:
    """Download recent kline data for ``stock_codes``.

    The returned dict maps each stock code to a :class:`pandas.DataFrame` with
    at most ``days`` of records. Data are cached locally via ``EastMoneyAPI``.
    """

    api = api or EastMoneyAPI()
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=days + 10)
    out: Dict[str, pd.DataFrame] = {}
    for code in stock_codes:
        df = api.get_kline_data(code, klt=klt, num=1000)
        if df is not None:
            df = df[df["date"] >= start_date.strftime("%Y-%m-%d")].reset_index(drop=True)
            out[code] = df
    return out
