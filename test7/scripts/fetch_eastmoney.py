"""
拉取指定股票近N日K线，保存到 data/raw/*.json
入参: --symbols, --beg, --end, --klt, --fqt
依赖: src.data.eastmoney_client
"""
import argparse


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    # 在这里添加参数，例如 symbols、beg、end 等
    return parser.parse_args()


def main():
    args = parse_args()
    # 根据参数调用 eastmoney_client.fetch_kline，并保存结果到文件
    raise NotImplementedError("此脚本需实现拉取数据并保存")


if __name__ == "__main__":
    main()
