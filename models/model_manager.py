from .wrapper_factory import WrapperFactory
from .base import BaseModelWrapper
import logging
import time
import gc
from .exceptions import ModelNotFoundException, ModelLoadException

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, model_configs):
        self.loaded_model: BaseModelWrapper = None
        self.wrapper_factory = WrapperFactory()
        self.model_configs = model_configs

    def load_model(self, model_name: str) -> tuple[bool, BaseModelWrapper]:
        if self.loaded_model is None or (self.loaded_model and self.loaded_model.model_name != model_name):
            logger.info(f"Attempting to load {model_name}")
            if self.loaded_model:
                loadedModelName = self.loaded_model.model_name
                logger.info(f"Attempting to unload {loadedModelName}")
                self.loaded_model.cleanup()
                self.loaded_model = None
                gc.collect()
                time.sleep(1)
                logger.info(f"Successfully unloaded {loadedModelName}")

            try:
                model_config = self.model_configs.get(model_name)
                if not model_config:
                    raise ValueError(f"Model {model_name} not found in configuration")
                logger.debug(f"Attempting to load {model_name} with config: {model_config}")
                new_model = self.wrapper_factory.get_wrapper(model_name, model_config)
                self.loaded_model = new_model
                logger.info(f"Successfully switched to {model_name}")
                return True, self.loaded_model
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {str(e)}")
                return False, None

        return True, self.loaded_model

    def get_current_model(self) -> BaseModelWrapper:
        return self.loaded_model

    def get_current_model_name(self) -> str:
        return self.loaded_model.model_name if self.loaded_model else None

    def is_model_loaded(self, model_name: str) -> bool:
        return self.loaded_model and self.loaded_model.model_name == model_name

    def switch_model(self, model_name: str) -> None:
        if model_name not in self.model_configs:
            logger.error(f"Model '{model_name}' not found in configurations")
            raise ModelNotFoundException(f"Model '{model_name}' not found")

        success, _ = self.load_model(model_name)
        if not success:
            logger.error(f"Failed to switch to model: {model_name}")
            raise ModelLoadException(f"Failed to load model: {model_name}")

        logger.info(f"Successfully switched to model: {model_name}")

    def get_model_configs(self):
        return self.model_configs
