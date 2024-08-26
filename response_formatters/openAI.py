from .base import BaseResponseFormatter
from typing import Any, Dict, List
import time
import uuid


class OpenAIResponseFormatter(BaseResponseFormatter):
    def format_response(self, raw_response: Any) -> Dict:
        """
        Format the raw response into an OpenAI-like structure for streaming.

        Args:
            raw_response (Any): The raw output from the model.

        Returns:
            Dict: Formatted response in OpenAI-like structure for streaming.
        """
        try:
            # Check if raw_response is a string (which might be the case for streaming responses)
            if isinstance(raw_response, str):
                content = raw_response
            elif isinstance(raw_response, dict) and 'choices' in raw_response:
                content = raw_response['choices'][0].get('text', '')
            else:
                raise ValueError(f"Unexpected raw_response format: {type(raw_response)}")

            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "llama",  # You might want to make this dynamic
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": content,
                        },
                        "finish_reason": None,
                    }
                ],
            }
        except Exception:
            # Return a minimal valid response to keep the stream going
            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "llama",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }

    def _format_choices(self, raw_choices: List[Dict]) -> List[Dict]:
        """Format the choices from the raw response for streaming."""
        return [
            {
                "index": i,
                "delta": {
                    "role": "assistant",
                    "content": choice.get('text', ''),  # Preserve all original formatting
                },
                "finish_reason": choice.get('finish_reason', None),
            }
            for i, choice in enumerate(raw_choices)
        ]
