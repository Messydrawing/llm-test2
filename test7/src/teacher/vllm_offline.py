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
    record_meta: bool = True,
) -> Iterable[Dict]:
    """统一的离线推理接口。

    优先尝试使用 vLLM 进行推理；若环境中缺少 vLLM 或 GPU 支持，
    则自动回退到 Hugging Face Transformers，保证最小功能可用。

    :return: 迭代返回 {"prompt": str, "output": str, "meta": {...}}
    """

    prompts = list(prompts)

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
        outputs = llm.generate(prompts, sampling)
        for prompt, out in zip(prompts, outputs):
            item = {"prompt": prompt, "output": out.outputs[0].text}
            if record_meta:
                item["meta"] = out.outputs[0].logprobs
            yield item
        return
    except Exception:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            device = (
                model.device
                if hasattr(model, "device")
                else next(model.parameters()).device
            )
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                output_ids = model.generate(
                    **inputs,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                )
                text = tokenizer.decode(
                    output_ids[0, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                yield {"prompt": prompt, "output": text}
        except Exception as e:
            raise RuntimeError(
                "vLLM 未安装或初始化失败，请检查依赖与 GPU 环境"
            ) from e


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
        allowed = schema.get("properties", {}).get("prediction", {}).get("enum", [])
        if cleaned["prediction"] not in allowed:
            cleaned["prediction"] = "flat"
        jsonschema.validate(cleaned, schema)
        return cleaned
