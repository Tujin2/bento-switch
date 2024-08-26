import unittest
from unittest.mock import patch
from response_formatters.openAI import OpenAIResponseFormatter


class TestOpenAIResponseFormatter(unittest.TestCase):
    def setUp(self):
        self.formatter = OpenAIResponseFormatter()

    def test_format_non_streaming_response(self):
        raw_response = {
            "model": "gpt-3.5-turbo",
            "choices": [{"text": "Hello, world!", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        formatted = self.formatter.format_non_streaming_response(raw_response)

        self.assertEqual(formatted["object"], "chat.completion")
        self.assertEqual(formatted["model"], "gpt-3.5-turbo")
        self.assertEqual(len(formatted["choices"]), 1)
        self.assertEqual(formatted["choices"][0]["message"]["content"], "Hello, world!")
        self.assertEqual(formatted["usage"], raw_response["usage"])

    @patch("time.time")
    def test_format_streaming_response(self, mock_time):
        mock_time.return_value = 1625097600  # July 1, 2021 00:00:00 UTC

        raw_response = {
            "model": "gpt-3.5-turbo",
            "choices": [{"text": "Hello", "finish_reason": None}],
        }

        formatted = self.formatter.format_streaming_response(raw_response)

        self.assertEqual(formatted["object"], "chat.completion.chunk")
        self.assertEqual(formatted["model"], "gpt-3.5-turbo")
        self.assertEqual(formatted["created"], 1625097600)
        self.assertEqual(formatted["choices"][0]["delta"]["content"], "Hello")
        self.assertIsNone(formatted["choices"][0]["finish_reason"])

        # Test final chunk
        raw_response["choices"][0]["text"] = ""
        raw_response["choices"][0]["finish_reason"] = "stop"

        formatted = self.formatter.format_streaming_response(raw_response)

        self.assertEqual(formatted["choices"][0]["delta"], {})
        self.assertEqual(formatted["choices"][0]["finish_reason"], "stop")

    def test_format_choices(self):
        raw_choices = [
            {"text": "Response 1", "finish_reason": "stop"},
            {"text": "Response 2", "finish_reason": "length"},
        ]

        formatted_choices = self.formatter._format_choices(raw_choices)

        self.assertEqual(len(formatted_choices), 2)
        self.assertEqual(formatted_choices[0]["message"]["content"], "Response 1")
        self.assertEqual(formatted_choices[0]["finish_reason"], "stop")
        self.assertEqual(formatted_choices[1]["message"]["content"], "Response 2")
        self.assertEqual(formatted_choices[1]["finish_reason"], "length")


if __name__ == "__main__":
    unittest.main()
