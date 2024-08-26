from pydantic import BaseModel
from typing import List, Optional
from .common import Message, GenerationParameters, UsageInfo
from openai.types.chat import ChatCompletionMessage


class ChatCompletionRequest(GenerationParameters):
    model: str
    messages: List[Message]


class ChatCompletionResponseChoice(BaseModel):
    index: Optional[int] = 0
    message: ChatCompletionMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
