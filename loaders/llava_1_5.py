from typing import Tuple

from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    PreTrainedTokenizer,
)

from . import register_loader
from .base import BaseModelLoader


@register_loader("llava-1.5")
class LLaVA15ModelLoader(BaseModelLoader):
    def load(
        self, load_model: bool = True
    ) -> Tuple[LlavaForConditionalGeneration, PreTrainedTokenizer, AutoProcessor]:
        if load_model:
            model = LlavaForConditionalGeneration.from_pretrained(
                self.model_path,
                **self.loading_kwargs,
            )
            model.config.hidden_size = (
                model.language_model.config.hidden_size
            )  # useful for deepspeed
        else:
            model = None

        processor = AutoProcessor.from_pretrained(self.model_path)
        tokenizer = processor.tokenizer
        return model, tokenizer, processor
