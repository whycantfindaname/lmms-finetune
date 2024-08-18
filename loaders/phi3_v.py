from typing import Tuple

from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from . import register_loader
from .base import BaseModelLoader


@register_loader("phi3-v")
class Phi3VModelLoader(BaseModelLoader):
    def load(
        self, load_model: bool = True
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer, AutoProcessor]:
        self.loading_kwargs["trust_remote_code"] = True
        model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **self.loading_kwargs,
            )
            if load_model
            else None
        )
        processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        tokenizer = processor.tokenizer
        return model, tokenizer, processor
