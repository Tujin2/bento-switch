from .base import BaseResponseFormatter
from typing import Any, Dict, List
import time
import uuid


class OpenAIResponseFormatter(BaseResponseFormatter):
    def __init__(self):
        self.current_stream_id = None
        self.creation_timestamp = None

    def format_response(self, raw_response: Any, streaming: bool = False) -> Dict:
        if streaming:
            return self.format_streaming_response(raw_response)
        else:
            return self.format_non_streaming_response(raw_response)

    def format_non_streaming_response(self, raw_response: Dict) -> Dict:
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": raw_response.get("model", "unknown"),
            "choices": self._format_choices(raw_response["choices"]),
            "usage": raw_response.get(
                "usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            ),
        }

    def format_streaming_response(self, raw_response: Any) -> Dict:
        if self.current_stream_id is None:
            self.current_stream_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            self.creation_timestamp = int(time.time())

        content = (
            raw_response["choices"][0]["text"]
            if isinstance(raw_response, dict) and "choices" in raw_response
            else raw_response
        )
        is_final_chunk = content == "" or (
            isinstance(raw_response, dict)
            and raw_response["choices"][0].get("finish_reason") is not None
        )

        delta = {"role": "assistant", "content": content} if not is_final_chunk else {}

        return {
            "id": self.current_stream_id,
            "object": "chat.completion.chunk",
            "created": self.creation_timestamp,
            "model": raw_response.get("model", "unknown"),
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": (
                        raw_response["choices"][0].get("finish_reason")
                        if is_final_chunk
                        else None
                    ),
                }
            ],
        }

    def _format_choices(self, raw_choices: List[Dict]) -> List[Dict]:
        """Format the choices from the raw response."""
        return [
            {
                "index": choice.get(
                    "index", i
                ),  # Use the provided index or fallback to the list index
                "message": {
                    "role": "assistant",
                    "content": choice.get("text", ""),
                },
                "finish_reason": choice.get("finish_reason", "stop"),
            }
            for i, choice in enumerate(raw_choices)
        ]
