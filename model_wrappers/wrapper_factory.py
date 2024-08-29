import yaml
from typing import Dict, Type
from .base import BaseModelWrapper
from .llama import LLaMAWrapper
from .flux import FluxWrapper
from constants import DEFAULT_N_CONTEXT, DEFAULT_N_GPU_LAYERS
import torch


class WrapperFactory:
    """
    A factory class for creating and managing model wrappers.

    This class provides a centralized way to create instances of different model wrappers
    based on the model type. It also allows for registration of new wrapper classes,
    making it extensible for future model types.

    Attributes:
        _wrappers (Dict[str, Type[BaseModelWrapper]]): A dictionary mapping model types
            to their corresponding wrapper classes.

    Usage:
        # Get a wrapper instance
        llama_wrapper = WrapperFactory.get_wrapper("llama", "llama-7b", "/path/to/model")

        # Register a new wrapper class
        WrapperFactory.register_wrapper("new_model", NewModelWrapper)
    """

    _wrappers: Dict[str, Type[BaseModelWrapper]] = {
        "llama": LLaMAWrapper,
        "flux": FluxWrapper,
        # Add more wrappers here as they are implemented
    }

    @classmethod
    def get_wrapper(cls, model_name: str) -> BaseModelWrapper:
        """
        Get the appropriate model wrapper based on the model type.

        Args:
            model_name (str): The name of the model.

        Returns:
            BaseModelWrapper: An instance of the appropriate model wrapper.

        Raises:
            ValueError: If the model type is not supported.
        """
        with open("model_configs.yaml", "r") as file:
            config = yaml.safe_load(file)

        model_config = config["models"].get(model_name)
        if not model_config:
            raise ValueError(f"Model {model_name} not found in configuration")

        wrapper_class = cls._wrappers.get(model_config["type"].lower())
        if not wrapper_class:
            raise ValueError(f"Unsupported model type: {model_config['type']}")

        if model_config["type"].lower() == "flux":
            wrapper = wrapper_class(
                model_path=model_config["path"],
                torch_dtype=getattr(torch, model_config.get("torch_dtype", "bfloat16")),
                enable_cpu_offload=model_config.get("enable_cpu_offload", True),
                auto_format=True,
            )
        else:
            wrapper = wrapper_class(
                model_path=model_config["path"],
                n_context=model_config.get("n_context", DEFAULT_N_CONTEXT),
                n_gpu_layers=model_config.get("n_gpu_layers", DEFAULT_N_GPU_LAYERS),
                prompt_template=model_config.get("prompt_template"),
                system_message_template=model_config.get("system_message_template"),
                conversation_message_template=model_config.get(
                    "conversation_message_template"
                ),
                auto_format=True,
            )

        return wrapper

    @classmethod
    def register_wrapper(cls, model_type: str, wrapper_class: Type[BaseModelWrapper]):
        """
        Register a new wrapper class for a given model type.

        Args:
            model_type (str): The type of the model.
            wrapper_class (Type[BaseModelWrapper]): The wrapper class to register.
        """
        cls._wrappers[model_type.lower()] = wrapper_class
