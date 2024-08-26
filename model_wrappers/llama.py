from typing import List, Any
from llama_cpp import Llama
from pydantic import BaseModel
from .base import BaseModelWrapper
import logging

logger = logging.getLogger("bentoml")


class Message(BaseModel):
    role: str
    content: str


class LLaMAWrapper(BaseModelWrapper):
    def load_model(self) -> Any:
        logger.debug(f"Initializing LLaMA model with path: {self.model_path}")
        try:
            return Llama(
                model_path=self.model_path,
                n_gpu_layers=-1,
                n_ctx=17000,
            )
        except Exception as e:
            logger.error(f"Error initializing LLaMA model: {e}")
            raise

    def create_prompt(self, messages: List[Message]) -> str:
        logger.debug(f"Creating prompt from {len(messages)} messages")
        if self.auto_format:
            return "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
        else:
            return messages[
                -1
            ].content  # Return only the last message if auto_format is False

    def get_response(self, prompt: str, **kwargs) -> Any:
        logger.debug(f"Generating response for prompt: {prompt[:50]}...")
        try:
            return self.model(prompt=prompt, **kwargs)
        except Exception as e:
            logger.error(f"Error in get_response method: {e}")
            raise

    def format_output(self, raw_output: Any) -> dict:
        logger.debug("Formatting model output")
        try:
            # Implement the formatting logic here
            # This is a placeholder implementation
            return {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": raw_output["choices"][0]["text"],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21,
                },
            }
        except Exception as e:
            logger.error(f"Error in format_output method: {e}")
            raise
