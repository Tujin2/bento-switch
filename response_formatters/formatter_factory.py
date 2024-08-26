from typing import Dict, Type
from .base import BaseResponseFormatter
from .openAI import OpenAIResponseFormatter


class FormatterFactory:
    _formatters: Dict[str, Type[BaseResponseFormatter]] = {
        "openai": OpenAIResponseFormatter,
    }

    @classmethod
    def get_formatter(cls, api_type: str) -> BaseResponseFormatter:
        """
        Get the appropriate response formatter based on the API type.

        Args:
            api_type (str): The type of API format to use (e.g., "openai").

        Returns:
            BaseResponseFormatter: An instance of the appropriate response formatter.

        Raises:
            ValueError: If an invalid api_type is provided.
        """
        formatter_class = cls._formatters.get(api_type.lower())
        if formatter_class is None:
            raise ValueError(f"Unsupported API type: {api_type}")
        return formatter_class()

    @classmethod
    def register_formatter(cls, api_type: str, formatter_class: Type[BaseResponseFormatter]):
        """
        Register a new formatter class for a given API type.

        Args:
            api_type (str): The type of API format.
            formatter_class (Type[BaseResponseFormatter]): The formatter class to register.
        """
        cls._formatters[api_type.lower()] = formatter_class
