"""
将 raw K 线转为 processed 样本，生成 prompt/summary/change
并时间滚动切分，导出 data/processed/*.jsonl 与 data/splits/*
"""
import argparse


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="配置文件路径")
    # TODO: 需要时可增加覆盖参数
    return parser.parse_args()


def main():
    args = parse_args()
    # TODO: 读取配置，处理 raw K 线并生成数据集
    raise NotImplementedError("需要实现数据集构建逻辑")


if __name__ == "__main__":
    main()
