"""时间处理工具"""
import datetime as dt


def now_str() -> str:
    """返回当前时间的字符串表示"""
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
