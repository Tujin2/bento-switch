from typing import List, Any
from llama_cpp import Llama
from .base import BaseModelWrapper
from api.schemas import Message
import logging
import gc

logger = logging.getLogger("bentoml")


class LLaMAWrapper(BaseModelWrapper):
    def __init__(
        self,
        model_path: str,
        n_context: int,
        n_gpu_layers: int,
        prompt_template: str = None,
        system_message_template: str = None,
        conversation_message_template: str = None,
        auto_format: bool = True,
    ):
        self.model_path = model_path
        self.n_context = n_context
        self.n_gpu_layers = n_gpu_layers
        super().__init__(
            model_name="llama", model_path=model_path, auto_format=auto_format
        )
        if prompt_template:
            self.set_prompt_template(prompt_template)
        if system_message_template:
            self.set_system_message_template(system_message_template)
        if conversation_message_template:
            self.set_conversation_message_template(conversation_message_template)
        self.model = None
        self.ctx = None

    def load_model(self) -> Llama:
        logger.debug(f"load_model called, self.n_gpu_layers: {self.n_gpu_layers}")
        logger.debug(f"Initializing LLaMA model with path: {self.model_path}")
        try:
            if self.model is None:
                self.model = Llama(
                    model_path=self.model_path,
                    n_gpu_layers=self.n_gpu_layers,
                    n_ctx=self.n_context,
                )
                self.ctx = self.model.ctx
            return self.model
        except Exception as e:
            logger.error(f"Error initializing LLaMA model: {e}")
            raise

    def cleanup(self):
        if self.model is not None:
            self.model = None      # Remove reference to the model
            self.ctx = None
        gc.collect()

    def create_prompt(self, messages: List[Message]) -> str:
        logger.debug(f"Creating prompt from {len(messages)} messages")
        try:
            system_prompt = next(
                (msg.content for msg in messages if msg.role == "system"), ""
            )
            formatted_system_prompt = self.system_message_template.format(
                system_prompt=system_prompt
            )

            conversation_history = "\n".join(
                self.conversation_message_template.format(
                    role=msg.role, content=msg.content
                )
                for msg in messages[-15:]
                if msg.role in {"user", "assistant"}
            )

            formatted_prompt = self.prompt_template.format(
                system_prompt=formatted_system_prompt,
                conversation_history=conversation_history,
            )

            logger.debug(
                f"Formatted prompt: {formatted_prompt[:500]}..."
            )  # Log first 100 chars
            return formatted_prompt
        except Exception as e:
            logger.error(f"Error in create_prompt: {str(e)}")
            raise ValueError(f"Failed to create prompt: {str(e)}")

    def get_response(self, prompt: str, **kwargs) -> Any:
        logger.debug(f"Generating response for prompt: {prompt[:50]}...")
        try:
            self.load_model()  # Ensure model is loaded
            return self.model(prompt=prompt, **kwargs)
        except Exception as e:
            logger.error(f"Error in get_response method: {e}")
            raise

    def format_output(self, raw_output: Any) -> dict:
        logger.debug("Formatting model output")
        try:
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
