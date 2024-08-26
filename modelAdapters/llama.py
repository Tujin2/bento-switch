# modelAdapters/llama.py

from typing import List
from llama_cpp import Llama
from openai.types.chat import (
    ChatCompletionMessage,
)
from utils.responses import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
)
import time
import uuid
import logging 

# Set up logging
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger = logging.getLogger("bentoml")
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)


class LlamaAdapter:
    def __init__(self, model_path):
        logger.debug(f"Initializing LlamaAdapter with model path: {model_path}")
        try:
            self.model = Llama(
                model_path=model_path, n_gpu_layers=-1, n_ctx=17000, max_tokens=15000
            )
            logger.debug("Model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise e

    def predict(self, prompt, max_tokens=50):
        logger.debug(
            f"Predict called with prompt: {prompt[:50]}... and max_tokens: {max_tokens}"
        )
        try:
            result = self.model(prompt=prompt, max_tokens=max_tokens)
            logger.debug("Model prediction successful")
            return result["choices"][0]["text"]
        except Exception as e:
            logger.error(f"Error in predict method: {e}")
            raise e

    def create_completion_openai(
        self, messages: List[ChatCompletionMessage], stream=False, **kwargs
    ):
        logger.debug(
            f"create_completion_openai called with {len(messages)} messages and stream={stream}"
        )
        try:
            prompt = self._format_messages(messages)
            logger.debug(f"Formatted prompt: {prompt[:50]}...")

            if stream:
                logger.debug("Streaming response mode enabled")
                responses = self.model(prompt=prompt, stream=True, **kwargs)
                for response in responses:
                    yield self._generate_streaming_response(response)

                # Generate a final chunk with finish_reason set to 'stop'
                final_chunk = {"choices": [{"text": "", "finish_reason": "stop"}]}
                yield self._generate_streaming_response(final_chunk, is_final=True)
            else:
                logger.debug("Single response mode enabled")
                response = self.model(prompt=prompt, stream=False, **kwargs)
                return self._generate_single_response(response)
        except Exception as e:
            logger.error(f"Error in create_completion_openai method: {e}")
            raise e

    def _format_messages(self, messages: List[ChatCompletionMessage]):
        try:
            logger.debug(f"Formatting {len(messages)} messages")
            return "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
        except Exception as e:
            logger.error(f"Error in _format_messages method: {e}")
            raise e

    def _generate_streaming_response(self, response, is_final=False):
        try:
            logger.debug(f"Processing a single streaming response chunk: {response}")
            content = response["choices"][0]["text"]

            delta = {"role": "assistant", "content": content}

            # Set finish_reason only if this is the final chunk
            finish_reason = "stop" if is_final else None

            choices = [{"index": 0, "delta": delta, "finish_reason": finish_reason}]

            return {
                "id": str(uuid.uuid4()),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "llama",
                "choices": choices,
            }
        except Exception as e:
            logger.error(f"Error in _generate_streaming_response method: {e}")
            raise e

    def _generate_single_response(self, response):
        try:
            logger.debug("Generating single response")
            choices = [
                ChatCompletionResponseChoice(
                    message=ChatCompletionMessage(
                        role="assistant", content=response["text"]
                    ),
                    finish_reason="stop",
                )
            ]
            usage = response.get("usage", {})
            return ChatCompletionResponse(
                id=str(uuid.uuid4()),
                object="chat.completion",
                created=int(time.time()),
                model="llama",
                choices=choices,
                usage=usage,
            )
        except Exception as e:
            logger.error(f"Error in _generate_single_response method: {e}")
            raise e
