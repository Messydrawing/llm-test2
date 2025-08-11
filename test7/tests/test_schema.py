"""测试数据结构定义是否可实例化"""
from src.data.schema import PromptItem, TeacherJSON, Record


def test_prompt_item_fields():
    item = PromptItem(stock_code="000001", summary="demo", change=0.0, prompt="test")
    assert item.stock_code == "000001"
