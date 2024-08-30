import unittest
from unittest.mock import patch, MagicMock
from models.llama import LLaMAWrapper
from api.schemas import Message


class TestLLaMAWrapper(unittest.TestCase):

    def setUp(self):
        self.mock_llama = patch("models.llama.Llama").start()
        self.mock_model = MagicMock()
        self.mock_llama.return_value = self.mock_model

        self.wrapper = LLaMAWrapper(
            model_path="/mock/path/to/llama/model",
            n_context=2048,
            n_gpu_layers=-1,
            auto_format=True,
        )
        self.wrapper.model_name = "llama-7b"
        self.wrapper.model = self.mock_model  # Set the mocked model directly

    def tearDown(self):
        patch.stopall()

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
