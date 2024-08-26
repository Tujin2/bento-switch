from pydantic import BaseModel
from typing import List, Optional
from .common import Message


class RawCompletionRequest(BaseModel):
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 50


class RawCompletionResponse(BaseModel):
    raw_output: dict
