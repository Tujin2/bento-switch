from abc import ABC, abstractmethod
from typing import Any


class BaseResponseFormatter(ABC):
    @abstractmethod
    def format_response(self, raw_response: Any) -> dict:
        """
        Format the raw response from the model into a standardized structure.

        Args:
            raw_response (Any): The raw output from the model.

        Returns:
            dict: Formatted response in a standardized structure.
        """
        pass
