import datetime
import requests
import pandas as pd
from .config import STOCK_CODES, SUMMARY_DAYS


class EastMoneyAPI:
    """Simple wrapper for EastMoney kline data."""

    KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

    def __init__(self) -> None:
        self.session = requests.Session()

    def _secid(self, code: str) -> str:
        return f"{'1' if code.startswith('6') else '0'}.{code}"

    def get_kline_data(
        self, stock_code: str, klt: int = 101, num: int = 1000
    ) -> pd.DataFrame | None:
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
            klines = js["data"]["klines"]
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
            return df
        except Exception as e:
            print(f"[EastMoneyAPI] {stock_code} fetch failed: {e}")
            return None


def get_recent_data(days: int = SUMMARY_DAYS) -> dict[str, pd.DataFrame]:
    """Download data for configured stocks."""

    api = EastMoneyAPI()
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=days + 10)
    stock_data = {}
    for code in STOCK_CODES:
        df = api.get_kline_data(code, num=1000)
        if df is not None:
            df = df[df["date"] >= start_date.strftime("%Y-%m-%d")]
            stock_data[code] = df
    return stock_data
