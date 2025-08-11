"""
使用 vLLM Offline Inference 批推老师输出（严格 JSON），
保存到 data/teacher_outputs/*.jsonl
生成参数来自 configs/teacher_infer.yaml
"""
import argparse


def parse_args():
    """解析配置文件路径。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="配置文件路径")
    return parser.parse_args()


def main():
    args = parse_args()
    # TODO: 读取 YAML 配置, 调用 vllm_offline.run_offline_infer 并写入结果
    raise NotImplementedError("需实现教师模型离线推理逻辑")


if __name__ == "__main__":
    main()
