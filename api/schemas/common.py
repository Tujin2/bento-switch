from pydantic import BaseModel
from typing import Optional


class GenerationParameters(BaseModel):
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stream: Optional[bool] = None


class Message(BaseModel):
    role: str
    content: str


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
