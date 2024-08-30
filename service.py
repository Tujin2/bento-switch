from __future__ import annotations

import logging

import bentoml
from fastapi import FastAPI

from models.model_manager import ModelManager
from models.exceptions import ModelNotFoundException, ModelLoadException
from response_formatters.formatter_factory import FormatterFactory
from utils.config_loader import load_model_configs
from api import create_chat_completion, create_raw_completion, switch_model

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model configuration
default_model_name, model_configs = load_model_configs()


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
        self.model_manager = ModelManager(model_configs)
        self.formatter = FormatterFactory.get_formatter("openai")
        # Load the default model
        try:
            self.model_manager.load_model(default_model_name)
        except (ModelNotFoundException, ModelLoadException) as e:
            logger.error(f"Failed to load default model: {str(e)}")

    create_chat_completion = create_chat_completion
    create_raw_completion = create_raw_completion
    switch_model = switch_model

    @app.get("/v1/models")
    def list_models(self):
        model_configs = self.model_manager.get_model_configs()
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
    def service_info(self):
        return f"Service is using model: {self.model_manager.get_current_model_name()}"
