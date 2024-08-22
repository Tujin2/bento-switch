# modelAdapters/llama.py

from llama_cpp import Llama


class LlamaAdapter:
    def __init__(self, model_path):
        # Load the GGUF model with the specified path
        self.model = Llama(model_path=model_path, n_gpu_layers=-1)

    def predict(self, prompt, max_tokens=50):
        # Perform inference using the model
        result = self.model(prompt=prompt, max_tokens=max_tokens)
        return result["choices"][0]["text"]
