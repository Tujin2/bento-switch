from pydantic import BaseModel
from typing import List
from .common import Message, GenerationParameters


class RawCompletionRequest(GenerationParameters):
    messages: List[Message]


class RawCompletionResponse(BaseModel):
    raw_output: dict
