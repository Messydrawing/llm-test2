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

## 东财接口说明
使用 `push2his.eastmoney.com/api/qt/stock/kline/get` 获取 K 线数据，核心参数包括 `secid`、`klt`、`fqt`、`fields1` 与 `fields2`，其含义可参照 AkShare/efinance 源码及文档。生产环境需遵守目标站点使用条款并做好频率限制。

## 使用说明
在运行整体代码前，务必先装配好DistillKit环境（通过[github](https://github.com/arcee-ai/DistillKit)获取）
指令`make -C test7 all`可一键执行蒸馏的全部过程。

若无法直接访问 HuggingFace，可在 `configs/teacher_infer.yaml` 中设置 `hf_endpoint`
为镜像站点（例如 `https://hf-mirror.com`）。脚本会自动读取该字段并
设置 `HF_ENDPOINT` 环境变量，从而通过镜像下载模型与权重。

若已提前下载好模型，也可在配置中填写 `model_path` 指向本地目录，
脚本会优先使用该路径加载模型，避免联网。

## 参考
- DistillKit: hidden/logits KD 及 DeepSpeed 示例。
- TRL GKDTrainer 文档。
- vLLM Offline Inference 指南。
- Transformers 生成与前缀约束。
- 教师模型 `Qwen/Qwen2.5-32B-Instruct` 模型卡与许可。
- KD 综述列表: Awesome-Knowledge-Distillation-of-LLMs。
