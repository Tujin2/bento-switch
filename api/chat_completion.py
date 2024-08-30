import typing as t
import json
from fastapi import HTTPException
from bentoml import api
from models.exceptions import ModelNotFoundException, ModelLoadException
from .schemas import ChatCompletionRequest, GenerationParameters
from utils.constants import (
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
    DEFAULT_STREAM,
)
from utils.config_loader import load_model_configs
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model configuration
default_model_name, model_defaults = load_model_configs()


@api(route="/v1/chat/completions", input_spec=ChatCompletionRequest)
async def create_chat_completion(self, **request: t.Any):
    model_name = request.get("model", self.model_manager.get_current_model_name())
    try:
        self.model_manager.switch_model(model_name)
    except ModelNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelLoadException as e:
        raise HTTPException(status_code=500, detail=str(e))

    model_wrapper = self.model_manager.get_current_model()

    # Reset the formatter's stream-specific attributes
    self.formatter.current_stream_id = None
    self.formatter.creation_timestamp = None

    model_specific_defaults = model_defaults.get(model_name, {}).get(
        "default_params", {}
    )

    # Merge request parameters with model-specific defaults and service-wide defaults
    generation_params = GenerationParameters(
        temperature=request.get(
            "temperature",
            model_specific_defaults.get("temperature", DEFAULT_TEMPERATURE),
        )
        or DEFAULT_TEMPERATURE,
        max_tokens=request.get(
            "max_tokens",
            model_specific_defaults.get("max_tokens", DEFAULT_MAX_TOKENS),
        )
        or DEFAULT_MAX_TOKENS,
        top_p=request.get(
            "top_p", model_specific_defaults.get("top_p", DEFAULT_TOP_P)
        )
        or DEFAULT_TOP_P,
        top_k=request.get(
            "top_k", model_specific_defaults.get("top_k", DEFAULT_TOP_K)
        )
        or DEFAULT_TOP_K,
        stream=request.get(
            "stream", model_specific_defaults.get("stream", DEFAULT_STREAM)
        )
        or DEFAULT_STREAM,
    )

    messages = request.get("messages", [])
    prompt = model_wrapper.create_prompt(messages)

    response = model_wrapper.get_response(
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
