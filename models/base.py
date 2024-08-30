from abc import ABC, abstractmethod
from typing import List, Any, Dict


class BaseModelWrapper(ABC):
    DEFAULT_PROMPT_TEMPLATE = "{system_prompt}\n\n{conversation_history}"
    DEFAULT_SYSTEM_MESSAGE_TEMPLATE = "System: {system_prompt}"
    DEFAULT_CONVERSATION_MESSAGE_TEMPLATE = "{role}: {content}"

    def __init__(self, model_name: str, model_path: str, default_params: Dict = None):
        self.model_name = model_name
        self.model_path = model_path
        self.prompt_template = self.DEFAULT_PROMPT_TEMPLATE
        self.system_message_template = self.DEFAULT_SYSTEM_MESSAGE_TEMPLATE
        self.conversation_message_template = self.DEFAULT_CONVERSATION_MESSAGE_TEMPLATE
        self.default_params = default_params or {}

    def initialize_model(self):
        """Initialize the model after all attributes are set."""
        self.model = self.load_model()

    def set_prompt_template(self, template: str):
        """Set the prompt template for the model."""
        self.prompt_template = template

    def set_system_message_template(self, template: str):
        """Set the system message template for the model."""
        self.system_message_template = template

    def set_conversation_message_template(self, template: str):
        """Set the conversation message template for the model."""
        self.conversation_message_template = template

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

    @abstractmethod
    def cleanup(self):
        """Clean up resources when unloading the model."""
        pass
