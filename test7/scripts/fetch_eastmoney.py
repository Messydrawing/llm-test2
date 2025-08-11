"""
拉取指定股票近N日K线，保存到 data/raw/*.json
入参: --symbols, --beg, --end, --klt, --fqt
依赖: src.data.eastmoney_client
"""
import argparse


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="配置文件路径")
    # TODO: 若需额外覆盖参数可在此添加
    return parser.parse_args()


def main():
    args = parse_args()
    # TODO: 读取配置并调用 eastmoney_client.fetch_kline 保存到 data/raw
    raise NotImplementedError("此脚本需实现拉取数据并保存")


if __name__ == "__main__":
    main()
