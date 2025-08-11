# Qwen32B 到 8B 蒸馏工程

## 目标
- 教师模型: `Qwen/Qwen2.5-32B-Instruct`
- 学生模型: `Qwen/Qwen2.5-7B/8B-Instruct`
- 任务: 基于 A 股 30 日 K 线数据, 生成包含 `prediction/analysis/advice` 三字段的结构化 JSON 输出。

## 流程概览
1. 抓取东方财富 K 线数据。
2. 生成样本摘要与涨跌幅, 构建训练/验证/测试集。
3. 使用 vLLM Offline Inference 批量生成教师输出。
4. 依次进行 DistillKit 隐藏态 KD、对数 KD, 再执行 TRL GKD 一轮 on-policy 蒸馏。
5. 统一评测并进行概率校准。

## 参考
- DistillKit: hidden/logits KD 及 DeepSpeed 示例。
- TRL GKDTrainer 文档。
- vLLM Offline Inference 指南。
- Transformers 生成与前缀约束。
- 教师模型 `Qwen/Qwen2.5-32B-Instruct` 模型卡与许可。
- KD 综述列表: Awesome-Knowledge-Distillation-of-LLMs。
