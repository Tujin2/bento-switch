from __future__ import annotations
import bentoml
from fastapi import FastAPI, Depends, HTTPException
import logging
import json
from response_formatters.formatter_factory import FormatterFactory
from model_wrappers.wrapper_factory import WrapperFactory
from api.schemas import (
    RawCompletionRequest,
    RawCompletionResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
import typing as t

model_path = "c:/models/bartowski/Codestral-22B-v0.1-GGUF/Codestral-22B-v0.1-Q6_K.gguf"


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
        # self.model = LlamaAdapter(model_path)
        self.model_wrapper = WrapperFactory.get_wrapper(
            "Codestral-22B-v0.1"
        )
        self.formatter = FormatterFactory.get_formatter("openai")
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

    # TODO: Update to use the new schemas
    @bentoml.api(route="/v1/chat/completions", input_spec=ChatCompletionRequest)
    async def create_chat_completion_stream(self, **request: t.Any):
        logger.info("Chat completion stream started")
        logger.debug(f"Received request: {request}")

        try:
            messages = request.get('messages', [])
            prompt = self.model_wrapper.create_prompt(messages)
            response_generator = self.model_wrapper.get_response(
                prompt,
                temperature=request.get("temperature", 0.8),
                max_tokens=request.get("max_tokens", 2000),
                top_p=request.get("top_p", 0.7),
                top_k=request.get("top_k", 0),
                stream=request.get("stream", True),
            )

            for raw_response in response_generator:
                logger.debug(f"Raw response from model: {raw_response}")
                try:
                    formatted_response = self.formatter.format_response(raw_response)
                    yield f"data: {json.dumps(formatted_response)}\n\n"
                except AttributeError as ae:
                    logger.error(f"AttributeError in formatting response: {str(ae)}")
                    logger.error(f"Raw response causing error: {raw_response}")
                    # Handle the error gracefully, possibly by skipping this response
                    continue
                except Exception as e:
                    logger.error(f"Error in formatting response: {str(e)}")
                    logger.error(f"Raw response causing error: {raw_response}")
                    # Handle other exceptions as needed
                    continue

            yield "data: [DONE]\n\n"  # Signal that streaming is complete
        except Exception as e:
            logger.error(f"Error in create_chat_completion_stream: {str(e)}")
            error_response = {"error": {"message": str(e), "type": "internal_error"}}
            yield f"data: {json.dumps(error_response)}\n\n"

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
