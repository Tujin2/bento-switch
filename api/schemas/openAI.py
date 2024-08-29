from pydantic import BaseModel, Field
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


class ImageGenerationRequest(BaseModel):
    model: str = Field(..., description="The model to use for image generation")
    prompt: str = Field(..., description="The prompt to generate the image from")
    n: int = Field(1, description="The number of images to generate", ge=1, le=10)
    size: str = Field("1024x1024", description="The size of the generated image(s)")
