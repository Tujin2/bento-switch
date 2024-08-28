from .wrapper_factory import WrapperFactory
from .base import BaseModelWrapper
import logging
import time
import gc

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self.loaded_model: BaseModelWrapper = None
        self.wrapper_factory = WrapperFactory()
        self.current_model_name: str = None

    def load_model(self, model_name: str) -> tuple[bool, BaseModelWrapper]:
        if self.current_model_name != model_name:
            logger.info(f"Attempting to switch from {self.current_model_name} to {model_name}")
            if self.loaded_model:
                self.loaded_model.cleanup()
                self.loaded_model = None
                gc.collect()
                time.sleep(1)

            try:
                new_model = self.wrapper_factory.get_wrapper(model_name)
                self.loaded_model = new_model
                self.current_model_name = model_name
                logger.info(f"Successfully switched to {model_name}")
                return True, self.loaded_model
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {str(e)}")
                return False, None

        return True, self.loaded_model

    def get_current_model(self) -> BaseModelWrapper:
        return self.loaded_model

    def get_current_model_name(self) -> str:
        return self.current_model_name

    def is_model_loaded(self, model_name: str) -> bool:
        return self.current_model_name == model_name
