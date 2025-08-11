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
    # 读取 YAML 配置, 调用 src.teacher.vllm_offline.run_offline_infer
    # 并将结果写入指定文件
    raise NotImplementedError("需实现教师模型离线推理逻辑")


if __name__ == "__main__":
    main()
