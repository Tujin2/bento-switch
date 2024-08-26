import unittest
from unittest.mock import patch, MagicMock
from model_wrappers.llama import LLaMAWrapper
from api.schemas import Message


class TestLLaMAWrapper(unittest.TestCase):

    @patch("model_wrappers.llama.Llama")
    def setUp(self, mock_llama):
        self.mock_model = MagicMock()
        mock_llama.return_value = self.mock_model
        self.wrapper = LLaMAWrapper("llama-7b", "/path/to/llama/model")

    def test_create_prompt(self):
        messages = [Message(role="user", content="Hello, how are you?")]
        prompt = self.wrapper.create_prompt(messages)
        self.assertIsInstance(prompt, str)
        self.assertIn("Hello, how are you?", prompt)

    def test_get_response(self):
        self.mock_model.return_value = {
            "choices": [{"text": "I'm doing well, thank you!"}]
        }
        prompt = "Hello, how are you?"
        response = self.wrapper.get_response(prompt)
        self.assertIn("choices", response)
        self.assertEqual(response["choices"][0]["text"], "I'm doing well, thank you!")
        self.mock_model.assert_called_once_with(prompt=prompt)

    def test_format_output(self):
        raw_output = {"choices": [{"text": "I'm doing well, thank you!"}]}
        formatted_output = self.wrapper.format_output(raw_output)
        self.assertIsInstance(formatted_output, dict)
        self.assertIn("choices", formatted_output)
        self.assertEqual(
            formatted_output["choices"][0]["message"]["content"],
            "I'm doing well, thank you!",
        )


if __name__ == "__main__":
    unittest.main()
