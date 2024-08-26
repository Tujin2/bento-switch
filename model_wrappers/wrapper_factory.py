from typing import Dict, Type
from model_wrappers.base import BaseModelWrapper
from model_wrappers.llama import LLaMAWrapper


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
        # Add more wrappers here as they are implemented
    }

    @classmethod
    def get_wrapper(
        cls, model_type: str, model_name: str, model_path: str
    ) -> BaseModelWrapper:
        """
        Get the appropriate model wrapper based on the model type.

        Args:
            model_type (str): The type of the model (e.g., "llama", "mistral").
            model_name (str): The name of the model.
            model_path (str): The path to the model file.

        Returns:
            BaseModelWrapper: An instance of the appropriate model wrapper.

        Raises:
            ValueError: If the model type is not supported.
        """
        wrapper_class = cls._wrappers.get(model_type.lower())
        if wrapper_class is None:
            raise ValueError(f"Unsupported model type: {model_type}")

        return wrapper_class(model_name=model_name, model_path=model_path)

    @classmethod
    def register_wrapper(cls, model_type: str, wrapper_class: Type[BaseModelWrapper]):
        """
        Register a new wrapper class for a given model type.

        Args:
            model_type (str): The type of the model.
            wrapper_class (Type[BaseModelWrapper]): The wrapper class to register.
        """
        cls._wrappers[model_type.lower()] = wrapper_class
