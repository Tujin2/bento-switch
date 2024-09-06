from .wrapper_factory import WrapperFactory
from .base import BaseModelWrapper
import logging
import time
import gc
from .exceptions import ModelNotFoundException, ModelLoadException
import threading

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, model_configs, keep_model_loaded=True, unload_delay_secs=0):
        self.loaded_model: BaseModelWrapper = None
        self.wrapper_factory = WrapperFactory()
        self.model_configs = model_configs
        self.model_unload_delay_secs = unload_delay_secs
        self.unload_timer = None
        self.last_use_time = 0
        self.mode = "keep_loaded"  # Default mode

    def load_model(self, model_name: str) -> tuple[bool, BaseModelWrapper]:
        if self.mode == "off":
            logger.info("Model loading is disabled (Off mode).")
            return False, None

        if self.loaded_model is None or (self.loaded_model and self.loaded_model.model_name != model_name):
            self._cancel_unload_timer()
            if self.loaded_model:
                self._unload_current_model()

            try:
                model_config = self.model_configs.get(model_name)
                if not model_config:
                    raise ValueError(f"Model {model_name} not found in configuration")
                logger.debug(f"Attempting to load {model_name} with config: {model_config}")
                new_model = self.wrapper_factory.get_wrapper(model_name, model_config)
                self.loaded_model = new_model
                logger.info(f"Successfully switched to {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {str(e)}")
                return False, None

        return True, self.loaded_model

    def _unload_current_model(self):
        if self.loaded_model:
            logger.info(f"Unloading model {self.loaded_model.model_name}")
            self.loaded_model.cleanup()
            self.loaded_model = None
            gc.collect()
            time.sleep(1)

    def schedule_unload(self):
        if self.mode == "dynamic":
            if self.unload_timer:
                self.unload_timer.cancel()

            if self.model_unload_delay_secs > 0:
                self.unload_timer = threading.Timer(self.model_unload_delay_secs, self._unload_current_model)
                self.unload_timer.start()
            else:
                self._unload_current_model()

    def _cancel_unload_timer(self):
        if self.unload_timer:
            self.unload_timer.cancel()
            self.unload_timer = None

    def update_last_use_time(self):
        self.last_use_time = time.time()
        self.schedule_unload()

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

    def set_mode(self, mode: str, timeout: int = 0):
        self.mode = mode
        if mode == "dynamic":
            self.model_unload_delay_secs = timeout
        elif mode == "off":
            self._unload_current_model()
        logger.info(f"ModelManager mode set to {mode} with timeout {timeout} seconds.")
