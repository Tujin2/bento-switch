from __future__ import annotations

import json
import logging
import typing as t

import bentoml
from fastapi import Depends, FastAPI, HTTPException

from api.schemas import (
    ChatCompletionRequest,
    RawCompletionRequest,
    RawCompletionResponse,
)
from api.schemas.common import GenerationParameters
from config_loader import load_model_configs
from constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_STREAM,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)
from model_wrappers.wrapper_factory import WrapperFactory
from response_formatters.formatter_factory import FormatterFactory

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model configuration
default_model_name, model_defaults = load_model_configs()


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
        self.model_id = default_model_name
        self.model_wrapper = WrapperFactory.get_wrapper(self.model_id)
        self.formatter = FormatterFactory.get_formatter("openai")
        self.model_id = default_model_name

    @bentoml.api(route="/v1/chat/completions", input_spec=ChatCompletionRequest)
    async def create_chat_completion(self, **request: t.Any):
        logger.info("Chat completion started")
        logger.debug(f"Received request: {request}")

        # Reset the formatter's stream-specific attributes
        self.formatter.current_stream_id = None
        self.formatter.creation_timestamp = None

        model_specific_defaults = model_defaults.get(self.model_id, {}).get(
            "default_params", {}
        )

        # Merge request parameters with model-specific defaults and service-wide defaults
        generation_params = GenerationParameters(
            temperature=request.get(
                "temperature",
                model_specific_defaults.get("temperature", DEFAULT_TEMPERATURE),
            ) or DEFAULT_TEMPERATURE,
            max_tokens=request.get(
                "max_tokens", model_specific_defaults.get("max_tokens", DEFAULT_MAX_TOKENS)
            ) or DEFAULT_MAX_TOKENS,
            top_p=request.get(
                "top_p", model_specific_defaults.get("top_p", DEFAULT_TOP_P)
            ) or DEFAULT_TOP_P,
            top_k=request.get(
                "top_k", model_specific_defaults.get("top_k", DEFAULT_TOP_K)
            ) or DEFAULT_TOP_K,
            stream=request.get(
                "stream", model_specific_defaults.get("stream", DEFAULT_STREAM)
            ) or DEFAULT_STREAM,
        )

        messages = request.get("messages", [])
        prompt = self.model_wrapper.create_prompt(messages)

        response = self.model_wrapper.get_response(
            prompt,
            temperature=generation_params.temperature,
            max_tokens=generation_params.max_tokens,
            top_p=generation_params.top_p,
            top_k=generation_params.top_k,
            stream=generation_params.stream,
        )

        logger.debug(f"Raw response from model: {response}")

        if generation_params.stream:
            for raw_response in response:
                logger.debug("Streaming response")

                try:
                    formatted_response = self.formatter.format_response(
                        raw_response, streaming=True
                    )
                    yield f"data: {json.dumps(formatted_response)}\n\n"
                except AttributeError as ae:
                    logger.error(f"AttributeError in formatting response: {str(ae)}")
                    logger.error(f"Raw response causing error: {raw_response}")
                    continue
                except Exception as e:
                    logger.error(f"Error in formatting response: {str(e)}")
                    logger.error(f"Raw response causing error: {raw_response}")
                    continue

            yield "data: [DONE]\n\n"  # Signal that streaming is complete
        else:
            logger.debug("Non-streaming response")
            formatted_response = self.formatter.format_response(
                response, streaming=False
            )
            yield formatted_response

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
    def list_models():
        _, model_configs = load_model_configs()
        models_list = [
            {
                "id": model_name,
                "object": "model",
                "created": 1677610602,
                "owned_by": "organization-owner",
            }
            for model_name in model_configs.keys()
        ]
        return {
            "object": "list",
            "data": models_list,
        }


@app.get("/service-info")
async def service_info(
    service: BentoSwitchService = Depends(bentoml.get_current_service),
):
    return f"Service is using model: {service.model_id}"
