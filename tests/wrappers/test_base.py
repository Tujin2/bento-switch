import unittest
from models.base import BaseModelWrapper


class DummyWrapper(BaseModelWrapper):
    def load_model(self):
        return "dummy model"

    def create_prompt(self, messages):
        return "dummy prompt"

    def get_response(self, prompt, **kwargs):
        return {"choices": [{"text": "dummy response"}]}

    def format_output(self, raw_output):
        return {"choices": [{"message": {"content": "dummy response"}}]}


class TestBaseModelWrapper(unittest.TestCase):

    def setUp(self):
        self.wrapper = DummyWrapper("dummy", "/path/to/dummy")

    def test_set_prompt_template(self):
        template = "Hello, {system_prompt} {prompt}"
        self.wrapper.set_prompt_template(template)
        self.assertEqual(self.wrapper.prompt_template, template)

    def test_create_prompt(self):
        messages = [{"role": "user", "content": "Hello"}]
        prompt = self.wrapper.create_prompt(messages)
        self.assertEqual(prompt, "dummy prompt")

    def test_get_response(self):
        response = self.wrapper.get_response("dummy prompt")
        self.assertEqual(response["choices"][0]["text"], "dummy response")

    def test_format_output(self):
        raw_output = {"choices": [{"text": "dummy response"}]}
        formatted_output = self.wrapper.format_output(raw_output)
        self.assertEqual(
            formatted_output["choices"][0]["message"]["content"], "dummy response"
        )


if __name__ == "__main__":
    unittest.main()
