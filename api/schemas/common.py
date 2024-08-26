from pydantic import BaseModel
from typing import Optional


class GenerationParameters(BaseModel):
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 50
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 0
    stream: Optional[bool] = True


class Message(BaseModel):
    role: str
    content: str


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
