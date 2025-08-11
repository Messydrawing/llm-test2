"""
JSON 合法率统计
"""


def count_valid(json_str_list) -> float:
    """计算给定 JSON 字符串列表的合法率"""
    import json

    if not json_str_list:
        return 0.0
    ok = 0
    for s in json_str_list:
        try:
            json.loads(s)
            ok += 1
        except Exception:
            pass
    return ok / len(json_str_list)
