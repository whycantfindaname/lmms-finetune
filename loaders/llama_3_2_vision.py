from typing import Tuple

from transformers import AutoProcessor, MllamaForConditionalGeneration, PreTrainedTokenizer, AutoConfig

from . import register_loader
from .base import BaseModelLoader


@register_loader("llama-3.2-vision")
class LLaMA32VisionModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True) -> Tuple[MllamaForConditionalGeneration, PreTrainedTokenizer, AutoProcessor, AutoConfig]:
        if load_model:
            model = MllamaForConditionalGeneration.from_pretrained(
                self.model_local_path, 
                **self.loading_kwargs,
            )
            model.config.hidden_size = model.language_model.config.hidden_size # useful for deepspeed
        else:
            model = None

        processor = AutoProcessor.from_pretrained(self.model_local_path, add_eos_token=True)
        tokenizer = processor.tokenizer
        config = AutoConfig.from_pretrained(self.model_local_path)
        return model, tokenizer, processor, config