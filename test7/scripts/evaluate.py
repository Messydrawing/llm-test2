"""
统一评测脚本:
  - 读取学生/老师在测试集的输出
  - 计算分类主指标（Acc/F1/MCC）与校准(Brier/ECE)
  - 统计 JSON 合法率
  - 产出与老师差距报告 (≤1~2pct 为达标线)
"""
import argparse


def parse_args():
    """解析评测配置。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="配置文件路径")
    return parser.parse_args()


def main():
    args = parse_args()
    # TODO: 根据配置读取数据与模型输出，计算各类指标
    raise NotImplementedError("需实现评测逻辑")


if __name__ == "__main__":
    main()
