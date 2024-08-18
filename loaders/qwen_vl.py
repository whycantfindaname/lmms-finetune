from typing import Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

from . import register_loader
from .base import BaseModelLoader


@register_loader("qwen-vl")
class QwenVLModelLoader(BaseModelLoader):
    def load(
        self, load_model: bool = True
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer, None]:
        self.loading_kwargs["trust_remote_code"] = True
        model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **self.loading_kwargs,
            )
            if load_model
            else None
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        return model, tokenizer, None
