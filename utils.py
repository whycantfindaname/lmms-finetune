import math
from collections import Counter
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import transformers
from deepspeed import zero
from torch.utils.data import Sampler
from transformers import Trainer
from transformers.trainer import has_length

class NoTextOnlyBatchSampler(Sampler):
    r"""
    Sampler that tries its best to sample batches such that no batch has only
    text (unimodal) data. This is necessary for training with deepspeed.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        is_text_only: Optional[List[bool]] = None,
        generator=None,
    ):
        if is_text_only is None:
            raise ValueError("`is_text_only` must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.is_text_only = is_text_only
        self.generator = generator
        self.mega_batch_size = batch_size * world_size

    def __len__(self):
        return len(self.is_text_only)

    def __iter__(self):
        # mm: multimodal, entry that has both text and image/video
        # uni: unimodal, entry that has only text
        mm_indices = [
            i for i, is_text_only in enumerate(self.is_text_only) if not is_text_only
        ]
        uni_indices = [
            i for i, is_text_only in enumerate(self.is_text_only) if is_text_only
        ]

        num_batches = math.ceil(
            (len(mm_indices) + len(uni_indices)) / self.mega_batch_size
        )
        if len(mm_indices) < num_batches:
            raise ValueError(
                f"{len(mm_indices)} multimodal entries, {len(num_batches)} batches. "
                "Not enough multimodal data in the dataset, or the batch size is too small. "
                "There will be at least one batch that is text-only, which doesn't work with deepspeed. "
                "Try increasing the batch size first."
            )

        # shuffle indices
        mm_indices = [
            mm_indices[i]
            for i in torch.randperm(len(mm_indices), generator=None).tolist()
        ]
        uni_indices = [
            uni_indices[i]
            for i in torch.randperm(len(uni_indices), generator=None).tolist()
        ]

        # distribute indices into batches
        num_uni_indices_in_mega_batch = [len(uni_indices) // num_batches] * num_batches
        for i in range(len(uni_indices) % num_batches):
            num_uni_indices_in_mega_batch[i] += 1

        mega_batches = []
        cur_uni_index = 0
        cur_mm_index = 0
        for i, num_uni_indices in enumerate(num_uni_indices_in_mega_batch):
            mega_batch = []
            mega_batch.extend(
                uni_indices[cur_uni_index : cur_uni_index + num_uni_indices]
            )
            cur_uni_index += num_uni_indices
            assert len(mega_batch) < self.mega_batch_size

            if i < num_batches - 1:
                increment = self.mega_batch_size - len(mega_batch)
                mega_batch.extend(mm_indices[cur_mm_index : cur_mm_index + increment])
                cur_mm_index += increment
            else:  # last batch
                mega_batch.extend(mm_indices[cur_mm_index:])
                assert len(mega_batch) <= self.mega_batch_size, "Last batch is too big."

            mega_batches.append(mega_batch)

        # mega_batch_indices = torch.randperm(len(mega_batches), generator=self.generator)
        # mega_batches = [mega_batches[i] for i in mega_batch_indices]
        indices = [i for mega_batch in mega_batches for i in mega_batch]
        return iter(indices)


class WeightedBatchSampler(Sampler):
    r"""
    Sampler that solves the class imbalance for an IQA dataset with optional custom weights.
    If weights are provided, those weights will be used directly for sampling.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        dataset: torch.utils.data.Dataset,
        generator=None,
    ):

        # WeightedBatchSampler is created at the beginning of the training process, 
        # the __init__ method will only run once when the instance is first created, 
        # not at the beginning of every epoch.

        if dataset is None:
            raise ValueError("`dataset` must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.generator = generator
        self.mega_batch_size = batch_size * world_size
        self.data_class = dataset.data_class
        self.weights = dataset.weights

        assert len(self.data_class) == len(self.weights), "Data class and weights must have the same length."

    def __len__(self):
        return 1
    # len(self.data_class) // self.mega_batch_size

    def __iter__(self):
        # 使用加权随机采样生成样本索引
        weighted_indices = torch.multinomial(
            self.weights, len(self.weights), replacement=True, generator=self.generator
        )
        self.weights[weighted_indices] *= 0.9
        # 将加权索引划分为mega batch
        num_batches = len(weighted_indices) // self.mega_batch_size
        mega_batches = [
            weighted_indices[
                i * self.mega_batch_size : (i + 1) * self.mega_batch_size
            ].tolist()
            for i in range(num_batches)
        ]

        # 如果剩余的样本不满一个mega batch，也作为一个批次
        if len(weighted_indices) % self.mega_batch_size != 0:
            mega_batches.append(
                weighted_indices[num_batches * self.mega_batch_size :].tolist()
            )

        # 展平mega_batches列表成为最终的样本索引序列
        indices = [i for mega_batch in mega_batches for i in mega_batch]
        iqa_indices = [i for i in indices if self.data_class[i] != "None"]
        unique_indices = list(set(indices))
        sampled_class = [self.data_class[i] for i in indices]
        iqa_unique_indices = list(set(iqa_indices))
        use_data_class = Counter(sampled_class).items()  
        
        rank0_print("Data class sampled in one epoch:", use_data_class)
        rank0_print("Number of unique data used in one epoch:", len(unique_indices))
        # rank0_print("Number of unique iqa data used in one epoch:", len(iqa_unique_indices))
        return iter(indices)


class TrainerWithCustomSampler(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        is_text_only = self.train_dataset.is_text_only
        iter_indices = NoTextOnlyBatchSampler(
            self.args.train_batch_size,
            world_size=self.args.world_size * self.args.gradient_accumulation_steps,
            is_text_only=is_text_only,
        )
        rank0_print(list(iter_indices))
        return iter_indices

    def _get_eval_sampler(
        self, eval_dataset: torch.utils.data.Dataset
    ) -> Optional[torch.utils.data.Sampler]:
        is_text_only = eval_dataset.is_text_only
        iter_indices = NoTextOnlyBatchSampler(
            self.args.eval_batch_size,
            world_size=self.args.world_size,
            is_text_only=is_text_only,
        )
        rank0_print(list(iter_indices))
        return iter_indices


class TrainerWithWeightedSampler(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        iter_indices = WeightedBatchSampler(
            self.args.train_batch_size,
            world_size=self.args.world_size * self.args.gradient_accumulation_steps,
            dataset=self.train_dataset,
        )
        rank0_print(list(iter_indices))
        return iter_indices

    def _get_eval_sampler(
        self, eval_dataset: torch.utils.data.Dataset
    ) -> Optional[torch.utils.data.Sampler]:
        is_text_only = eval_dataset.is_text_only
        iter_indices = NoTextOnlyBatchSampler(
            self.args.eval_batch_size,
            world_size=self.args.world_size,
            is_text_only=is_text_only,
        )
        rank0_print(list(iter_indices))
        return iter_indices
    
    # def _get_eval_sampler(
    #     self, eval_dataset: torch.utils.data.Dataset
    # ) -> Optional[torch.utils.data.Sampler]:
    #     # Should also use weighted sample strategy during evaluation
    #     weights = eval_dataset.weights
    #     rank0_print("Data class weights:", weights)
    #     iter_indices = NoTextOnlyBatchSampler(
    #         self.args.eval_batch_size,
    #         world_size=self.args.world_size,
    #         dataset=eval_dataset,
    #     )
    #     rank0_print(list(iter_indices))
    #     return iter_indices


def find_all_linear_names(named_modules: Dict, target_modules: List[str]):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in named_modules.items():
        if not any([module_name in name for module_name in target_modules]):
            continue

        if isinstance(module, cls):
            lora_module_names.add(name)

    for name in list(lora_module_names):
        if "lm_head" in name:  # needed for 16-bit
            lora_module_names.remove(name)

    return list(lora_module_names)


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args)


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)
