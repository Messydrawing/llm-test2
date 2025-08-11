"""
vLLM Offline Inference 批量生成老师答案（严格 JSON）
参考: vLLM Offline Inference 文档
"""
from typing import Iterable, Dict
import json


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
    try:
        from vllm import LLM, SamplingParams

        sampling = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop=stop,
        )
        llm = LLM(
            model=model_name,
            tensor_parallel_size=tp_size,
            max_model_len=max_model_len,
        )
        outputs = llm.generate(list(prompts), sampling)
        for prompt, out in zip(prompts, outputs):
            item = {"prompt": prompt, "output": out.outputs[0].text}
            if record_meta:
                item["meta"] = out.outputs[0].logprobs
            yield item
    except Exception:
        for p in prompts:
            yield {"prompt": p, "output": "", "meta": {}}


def ensure_json(output_str: str, schema_path: str) -> Dict:
    """
    校验/清洗老师输出为合法 JSON（必要时做小范围修复）。
    schema 可用 json_schema/teacher_output.schema.json
    """
    import jsonschema

    try:
        data = json.loads(output_str)
    except json.JSONDecodeError:
        start = output_str.find("{")
        end = output_str.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(output_str[start : end + 1])
        else:
            data = {}
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    try:
        jsonschema.validate(data, schema)
        return data
    except jsonschema.ValidationError:
        cleaned = {
            "prediction": data.get("prediction", ""),
            "analysis": data.get("analysis", ""),
            "advice": data.get("advice", ""),
        }
        jsonschema.validate(cleaned, schema)
        return cleaned
