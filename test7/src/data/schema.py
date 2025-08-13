from pydantic import BaseModel, Field
from typing import Literal, Optional


class PromptItem(BaseModel):
    stock_code: str
    kline_json: list[dict]
    change: float
    prompt: str  # 套用统一模板的最终提示串


class TeacherJSON(BaseModel):
    prediction: Literal["up","down","flat"]
    analysis: str
    advice: str


class Record(BaseModel):
    prompt: PromptItem
    teacher: Optional[TeacherJSON] = None
    meta: dict = Field(default_factory=dict)  # 采样温度/种子/时间戳等
