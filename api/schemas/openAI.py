from pydantic import BaseModel
from typing import List
from .common import Message, GenerationParameters


class ChatCompletionRequest(GenerationParameters):
    model: str
    messages: List[Message]


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict
