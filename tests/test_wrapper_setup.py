import unittest
from unittest.mock import patch, MagicMock
from model_wrappers import WrapperFactory
from model_wrappers.base import BaseModelWrapper
from model_wrappers.llama import LLaMAWrapper


class TestWrapperSetup(unittest.TestCase):

    @patch("model_wrappers.llama.Llama")
    def test_wrapper_factory_creation(self, mock_llama):
        # Test creation of LLaMA wrapper
        wrapper = WrapperFactory.get_wrapper(
            "llama", "llama-7b", "/path/to/llama/model"
        )
        self.assertIsInstance(wrapper, LLaMAWrapper)
        self.assertIsInstance(wrapper, BaseModelWrapper)
        mock_llama.assert_called_once_with(
            model_path="/path/to/llama/model", n_gpu_layers=-1, n_ctx=17000
        )

    def test_wrapper_factory_invalid_type(self):
        # Test error handling for invalid model type
        with self.assertRaises(ValueError):
            WrapperFactory.get_wrapper("invalid_model", "model", "/path/to/model")

    @patch("model_wrappers.llama.Llama")
    def test_llama_wrapper_methods(self, mock_llama):
        # Mock the Llama model
        mock_model = MagicMock()
        mock_llama.return_value = mock_model
        mock_model.return_value = {"choices": [{"text": "I'm doing well, thank you!"}]}

        wrapper = WrapperFactory.get_wrapper(
            "llama", "llama-7b", "/path/to/llama/model"
        )

        # Test create_prompt
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        prompt = wrapper.create_prompt(messages)
        self.assertIsInstance(prompt, str)
        self.assertIn("Hello, how are you?", prompt)

        # Test get_response
        response = wrapper.get_response(prompt)
        self.assertIn("choices", response)
        self.assertEqual(response["choices"][0]["text"], "I'm doing well, thank you!")
        mock_model.assert_called_once_with(prompt=prompt)

        # Test format_output
        formatted_output = wrapper.format_output(response)
        self.assertIsInstance(formatted_output, dict)
        self.assertIn("choices", formatted_output)
        self.assertEqual(
            formatted_output["choices"][0]["message"]["content"],
            "I'm doing well, thank you!",
        )


if __name__ == "__main__":
    unittest.main()
