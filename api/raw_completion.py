from fastapi import HTTPException
from bentoml import api
from models.exceptions import ModelNotFoundException, ModelLoadException
from .schemas import RawCompletionRequest, RawCompletionResponse
from utils.config_loader import load_model_configs
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model configuration
default_model_name, model_defaults = load_model_configs()


@api(route="/v1/raw_completion")
async def create_raw_completion(
    self, request: RawCompletionRequest
) -> RawCompletionResponse:
    try:
        model_name = (
            request.model
            if hasattr(request, "model")
            else self.model_manager.get_current_model_name()
        )
        self.model_manager.switch_model(model_name)
    except ModelNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelLoadException as e:
        raise HTTPException(status_code=500, detail=str(e))

    model_wrapper = self.model_manager.get_current_model()

    prompt = model_wrapper.create_prompt(request.messages)
    raw_output = model_wrapper.get_response(
        prompt,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    logger.info("Raw completion successful")
    return RawCompletionResponse(raw_output=raw_output)
