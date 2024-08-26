import unittest
from unittest.mock import patch, MagicMock
from model_wrappers import WrapperFactory
from model_wrappers.base import BaseModelWrapper
from model_wrappers.llama import LLaMAWrapper


class TestWrapperFactory(unittest.TestCase):

    @patch("model_wrappers.llama.Llama")
    @patch(
        "builtins.open",
        new_callable=unittest.mock.mock_open,
        read_data="""
    models:
      llama-7b:
        type: llama
        path: /path/to/llama/model
      dummy:
        type: dummy
        path: /path/to/dummy/model
    """,
    )
    def test_wrapper_factory_creation(self, mock_open, mock_llama):
        # Test creation of LLaMA wrapper
        wrapper = WrapperFactory.get_wrapper("llama-7b")
        self.assertIsInstance(wrapper, LLaMAWrapper)
        self.assertIsInstance(wrapper, BaseModelWrapper)
        mock_llama.assert_called_once_with(
            model_path="/path/to/llama/model", n_gpu_layers=-1, n_ctx=17000
        )

    @patch(
        "builtins.open",
        new_callable=unittest.mock.mock_open,
        read_data="""
    models:
      llama-7b:
        type: llama
        path: /path/to/llama/model
    """,
    )
    def test_wrapper_factory_invalid_type(self, mock_open):
        # Test error handling for invalid model type
        with self.assertRaises(ValueError):
            WrapperFactory.get_wrapper("invalid_model")

    @patch("model_wrappers.llama.Llama")
    @patch(
        "builtins.open",
        new_callable=unittest.mock.mock_open,
        read_data="""
    models:
      llama-7b:
        type: llama
        path: /path/to/llama/model
      dummy:
        type: dummy
        path: /path/to/dummy/model
    """,
    )
    def test_register_wrapper(self, mock_open, mock_llama):
        class DummyWrapper(BaseModelWrapper):
            def load_model(self):
                return MagicMock()

            def create_prompt(self, messages):
                return "dummy prompt"

            def get_response(self, prompt, **kwargs):
                return {"choices": [{"text": "dummy response"}]}

            def format_output(self, raw_output):
                return {"choices": [{"message": {"content": "dummy response"}}]}

        WrapperFactory.register_wrapper("dummy", DummyWrapper)
        wrapper = WrapperFactory.get_wrapper("dummy")
        self.assertIsInstance(wrapper, DummyWrapper)


if __name__ == "__main__":
    unittest.main()
