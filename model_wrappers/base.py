from abc import ABC, abstractmethod
from typing import List, Any


class BaseModelWrapper(ABC):
    def __init__(self, model_name: str, model_path: str, auto_format: bool = True):
        self.model_name = model_name
        self.model_path = model_path
        self.auto_format = auto_format
        self.model = self.load_model()
        self.prompt_template = ""

    def set_prompt_template(self, template: str):
        """Set the prompt template for the model."""
        self.prompt_template = template

    @abstractmethod
    def load_model(self) -> Any:
        """Load and return the model."""
        pass

    @abstractmethod
    def create_prompt(self, messages: List[dict]) -> str:
        """
        Generate the prompt based on the provided messages.

        Args:
            messages (List[dict]): List of message dictionaries.

        Returns:
            str: Formatted prompt.
        """
        pass

    @abstractmethod
    def get_response(self, prompt: str, **kwargs) -> Any:
        """
        Generate a response from the model based on the prompt and additional parameters.

        Args:
            prompt (str): The input prompt.
            **kwargs: Additional parameters like temperature, top_p, etc.

        Returns:
            Any: The raw model output.
        """
        pass

    @abstractmethod
    def format_output(self, raw_output: Any) -> dict:
        """
        Format the raw model output into the desired structure.

        Args:
            raw_output (Any): The raw output from the model.

        Returns:
            dict: Formatted output (e.g., OpenAI-like structure).
        """
        pass
