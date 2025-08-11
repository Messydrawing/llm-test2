"""
将 raw K 线转为 processed 样本，生成 prompt/summary/change
并时间滚动切分，导出 data/processed/*.jsonl 与 data/splits/*
"""
import argparse


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    # 可添加配置文件路径等参数
    return parser.parse_args()


def main():
    args = parse_args()
    # 读取 raw 数据, 调用 summarize_kline 等模块, 最终写入 JSONL
    raise NotImplementedError("需要实现数据集构建逻辑")


if __name__ == "__main__":
    main()
