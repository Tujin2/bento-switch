import unittest
from unittest.mock import patch, MagicMock
from models import WrapperFactory
from models.base import BaseModelWrapper


class TestWrapperFactory(unittest.TestCase):

    @patch.dict(
        "models.wrapper_factory.WrapperFactory._wrappers", {"llama": MagicMock()}
    )
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
    def test_wrapper_factory_creation(self, mock_open):
        mock_llama_wrapper = WrapperFactory._wrappers["llama"]

        # Test creation of LLaMA wrapper
        wrapper = WrapperFactory.get_wrapper("llama-7b")

        mock_llama_wrapper.assert_called_once_with(
            model_path="/path/to/llama/model",
            n_gpu_layers=-1,
            n_context=2048,
            prompt_template=None,
            system_message_template=None,
            conversation_message_template=None,
            auto_format=True,
        )
        self.assertEqual(wrapper, mock_llama_wrapper.return_value)

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

    @patch("models.llama.Llama")
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
            def __init__(self, model_path, **kwargs):
                self.model_path = model_path

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
