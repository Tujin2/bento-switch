import torch
from diffusers import FluxPipeline
from typing import Any
from .base import BaseModelWrapper
import logging
import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger("bentoml")


class FluxWrapper(BaseModelWrapper):
    def __init__(
        self,
        model_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        enable_cpu_offload: bool = True,
        auto_format: bool = True,
    ):
        super().__init__(
            model_name="flux", model_path=model_path, auto_format=auto_format
        )
        self.torch_dtype = torch_dtype
        self.enable_cpu_offload = enable_cpu_offload
        self.pipe = None

        # Authenticate with Hugging Face
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            login(token=hf_token)
        else:
            logger.warning("HUGGINGFACE_TOKEN not found in environment variables. You may not be able to download private models.")

    def load_model(self) -> FluxPipeline:
        logger.debug(f"Initializing Flux model with path: {self.model_path}")
        try:
            self.pipe = FluxPipeline.from_pretrained(
                self.model_path, torch_dtype=self.torch_dtype, use_auth_token=True
            )
            self.pipe.enable_model_cpu_offload()
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()

            return self.pipe
        except Exception as e:
            logger.error(f"Error initializing Flux model: {e}")
            raise

    def create_prompt(self, messages: list[dict]) -> str:
        # For image generation, we'll use the last user message as the prompt
        return messages[-1]["content"] if messages else ""

    def get_response(self, prompt: str, **kwargs) -> Any:
        logger.debug(f"Generating image for prompt: {prompt[:50]}...")
        try:
            self.load_model()  # Ensure model is loaded
            image = self.pipe(
                prompt,
                height=kwargs.get("height", 1024),
                width=kwargs.get("width", 1024),
                guidance_scale=kwargs.get("guidance_scale", 3.5),
                num_inference_steps=kwargs.get("num_inference_steps", 50),
                max_sequence_length=kwargs.get("max_sequence_length", 512),
                generator=torch.Generator("cpu").manual_seed(kwargs.get("seed", 0)),
            ).images[0]
            return image
        except Exception as e:
            logger.error(f"Error in get_response method: {e}")
            raise

    def format_output(self, raw_output: Any) -> dict:
        logger.debug("Formatting model output")
        try:
            return {
                "id": "imggen-123",
                "object": "image.generation",
                "created": 1677652288,
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "image": raw_output,
                        "finish_reason": "stop",
                    }
                ],
            }
        except Exception as e:
            logger.error(f"Error in format_output method: {e}")
            raise

    def cleanup(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        torch.cuda.empty_cache()
