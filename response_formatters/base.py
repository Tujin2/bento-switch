from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseResponseFormatter(ABC):
    @abstractmethod
    def format_response(self, raw_response: Any, streaming: bool = False) -> Dict:
        """
        Format the raw response from the model into a standardized structure.

        Args:
            raw_response (Any): The raw output from the model.
            streaming (bool): Whether this is a streaming response or not.

        Returns:
            dict: Formatted response in a standardized structure.
        """
        pass
