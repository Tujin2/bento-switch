from .base import BaseResponseFormatter
from typing import Any, Dict, List
import time
import uuid


class OpenAIResponseFormatter(BaseResponseFormatter):
    def format_response(self, raw_response: Any) -> Dict:
        """
        Format the raw response into an OpenAI-like structure.

        Args:
            raw_response (Any): The raw output from the model.

        Returns:
            Dict: Formatted response in OpenAI-like structure.
        """
        # Assuming raw_response is a dictionary with a 'choices' key
        if not isinstance(raw_response, dict) or 'choices' not in raw_response:
            raise ValueError("Invalid raw response format")

        choices = self._format_choices(raw_response['choices'])

        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": raw_response.get('model', 'unknown'),
            "choices": choices,
            "usage": self._calculate_usage(raw_response),
        }

    def _format_choices(self, raw_choices: List[Dict]) -> List[Dict]:
        """Format the choices from the raw response."""
        return [
            {
                "index": i,
                "message": {
                    "role": "assistant",
                    "content": choice.get('text', '').strip(),
                },
                "finish_reason": choice.get('finish_reason', 'stop'),
            }
            for i, choice in enumerate(raw_choices)
        ]

    def _calculate_usage(self, raw_response: Dict) -> Dict:
        """Calculate token usage (placeholder implementation)."""
        # This is a simplified version. In a real scenario, you'd need to implement
        # proper token counting logic.
        total_tokens = sum(len(choice.get('text', '').split()) for choice in raw_response['choices'])
        return {
            "prompt_tokens": 0,  # You'd need to calculate this based on the input
            "completion_tokens": total_tokens,
            "total_tokens": total_tokens,
        }
