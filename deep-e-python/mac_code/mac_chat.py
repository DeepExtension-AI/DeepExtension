from typing import List, Dict, Optional, Union, Generator
from mlx_lm import load, stream_generate
import os

class ModelInference:
    def __init__(self, model_path: str, adapter_path: str):
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        try:
            return load(self.model_path, adapter_path=self.adapter_path,tokenizer_config={"eos_token": "<|endoftext|>", "trust_remote_code": True})
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def generate_response(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 512,
       
    ) -> Generator[str, None, None]:
        """
        Generate model response (streaming output)

        Args:
            prompt: Raw text prompt (takes precedence over messages)
            messages: Chat history/messages
            max_tokens: Maximum number of tokens to generate

        Yields:
            str: Incrementally generated response text
        """
        try:
            final_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
        except Exception as e:
            raise ValueError(f"Failed to generate chat template: {str(e)}")

        for response in stream_generate(self.model, self.tokenizer, final_prompt, max_tokens=max_tokens):
            yield response.text