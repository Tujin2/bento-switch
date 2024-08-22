from __future__ import annotations
import bentoml
from modelAdapters import LlamaAdapter
from pydantic import BaseModel
from fastapi import FastAPI, Depends, HTTPException
from typing import List, Optional
import logging
import uuid
import time

# Define the path to your GGUF model
model_path = "c:/models/bartowski/Codestral-22B-v0.1-GGUF/Codestral-22B-v0.1-Q6_K.gguf"


# Define Pydantic models for the input
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 50


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict


# Create a FastAPI app instance
app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@bentoml.service(
    resources={"cpu": "14", "memory": "48Gi"},
    traffic={"timeout": 10},
)
@bentoml.mount_asgi_app(app, path="/v1")
class BentoSwitchService:
    def __init__(self):
        # Initialize the LlamaAdapter model
        self.model = LlamaAdapter(model_path)
        self.model_id = "Codestral-22B-v0.1"

    @app.post("/v1/chat/completions")
    async def create_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            # Combine all messages into a single prompt
            prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])

            # Generate text using the model
            logger.info(f"Generating text for prompt: {prompt[:50]}...")
            generated_text = self.model.predict(prompt=prompt, max_tokens=request.max_tokens)
            logger.info(f"Generated text: {generated_text[:50]}...")

            response = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4()}",
                created=int(time.time()),
                model=self.model_id,
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                usage={
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(generated_text.split()),
                    "total_tokens": len(prompt.split()) + len(generated_text.split()),
                },
            )
            logger.info("Response created successfully")
            return response
        except Exception as e:
            logger.error(f"Error in create_chat_completion: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1//models")
    def list_models(self):
        return {
            "object": "list",
            "data": [
                {
                    "id": self.model_id,
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": "organization-owner",
                }
            ]
        }


# If you need to access the service instance outside the class
@app.get("/service-info")
async def service_info(service: BentoSwitchService = Depends(bentoml.get_current_service)):
    return f"Service is using model: {service.model_id}"
