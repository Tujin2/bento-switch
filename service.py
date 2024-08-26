from __future__ import annotations
import bentoml
from modelAdapters import LlamaAdapter
from pydantic import BaseModel
from fastapi import FastAPI, Depends, HTTPException
from typing import List, Optional
import logging
import json
# from model_wrappers import WrapperFactory
from api.schemas import Message, RawCompletionRequest, RawCompletionResponse

model_path = "c:/models/bartowski/Codestral-22B-v0.1-GGUF/Codestral-22B-v0.1-Q6_K.gguf"


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


app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@bentoml.service(
    resources={"cpu": "14", "memory": "48Gi"},
    traffic={"timeout": 10},
    logging={
        "access": {
            "enabled": True,
            "request_content_length": True,
            "request_content_type": True,
            "response_content_length": True,
            "response_content_type": True,
            "skip_paths": ["/metrics", "/healthz", "/livez", "/readyz"],
            "format": {"trace_id": "032x", "span_id": "016x"},
        }
    },
)
@bentoml.mount_asgi_app(app, path="/")
class BentoSwitchService:
    def __init__(self):
        # Initialize the LlamaAdapter model
        self.model = LlamaAdapter(model_path)
        # self.new_wrapper = WrapperFactory.get_wrapper("llama", "Codestral-22B-v0.1", model_path)
        self.model_id = "Codestral-22B-v0.1"

    @app.post("/v1/chat/completions_nostream")
    async def create_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        try:
            response = self.model.create_completion_openai(
                request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            logger.info("Chat completion successful")
            return response
        except Exception as e:
            logger.error(f"Error in create_chat_completion: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @bentoml.api(route="/v1/chat/completions")
    def create_chat_completion_stream(
        self,
        model: str,
        messages: List[Message],
        temperature: Optional[float] = 0.9,
        max_tokens: Optional[int] = 1000,
    ):
        logger.info("Chat completion stream started")
        logger.debug(f"Received model: {model}")
        logger.debug(f"Received messages: {messages}")
        logger.debug(f"Received temperature: {temperature}")
        logger.debug(f"Received max_tokens: {max_tokens}")

        try:
            response_generator = self.model.create_completion_openai(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            for response in response_generator:
                yield f"data: {json.dumps(response)}\n\n"

            yield "data: [DONE]\n\n"  # Signal that streaming is complete
        except Exception as e:
            logger.error(f"Error in create_chat_completion_stream: {str(e)}")
            yield f"data: {str(e)}\n\n"

    @bentoml.api(route="/v1/raw_completion")
    async def create_raw_completion(
        self, request: RawCompletionRequest
    ) -> RawCompletionResponse:
        try:
            prompt = self.new_wrapper.create_prompt(request.messages)
            raw_output = self.new_wrapper.get_response(
                prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            logger.info("Raw completion successful")
            return RawCompletionResponse(raw_output=raw_output)
        except Exception as e:
            logger.error(f"Error in create_raw_completion: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/models")
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
            ],
        }


@app.get("/service-info")
async def service_info(
    service: BentoSwitchService = Depends(bentoml.get_current_service),
):
    return f"Service is using model: {service.model_id}"
