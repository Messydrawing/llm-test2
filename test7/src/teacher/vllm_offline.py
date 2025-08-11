"""
vLLM Offline Inference 批量生成老师答案（严格 JSON）
参考: vLLM Offline Inference 文档
"""
from typing import Iterable, Dict


def run_offline_infer(
    model_name: str,
    prompts: Iterable[str],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stop: list,
    tp_size: int = 1,
    max_model_len: int = 32768,
    record_meta: bool = True
) -> Iterable[Dict]:
    """
    :return: 迭代返回 { "prompt": str, "output": str, "meta": {...} }
    """
    # TODO: 调用 vLLM 离线推理接口，对批量 prompt 生成结果
    raise NotImplementedError


def ensure_json(output_str: str, schema_path: str) -> Dict:
    """
    校验/清洗老师输出为合法 JSON（必要时做小范围修复）。
    schema 可用 json_schema/teacher_output.schema.json
    """
    # TODO: 依据 JSON Schema 验证并必要时修正输出
    raise NotImplementedError
