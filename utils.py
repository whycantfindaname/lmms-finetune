import math
from collections import Counter
from typing import Dict, List, Optional
import logging
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
        mm_indices = [i for i, is_text_only in enumerate(self.is_text_only) if not is_text_only]
        uni_indices = [i for i, is_text_only in enumerate(self.is_text_only) if is_text_only]

        num_batches = math.ceil((len(mm_indices) + len(uni_indices)) / self.mega_batch_size)
        if len(mm_indices) < num_batches:
            raise ValueError(
                f"{len(mm_indices)} multimodal entries, {len(num_batches)} batches. "
                "Not enough multimodal data in the dataset, or the batch size is too small. " 
                "There will be at least one batch that is text-only, which doesn't work with deepspeed. "
                "Try increasing the batch size first."
            )

        # shuffle indices
        mm_indices = [mm_indices[i] for i in torch.randperm(len(mm_indices), generator=None).tolist()]
        uni_indices = [uni_indices[i] for i in torch.randperm(len(uni_indices), generator=None).tolist()]

        # distribute indices into batches
        num_uni_indices_in_mega_batch = [len(uni_indices) // num_batches] * num_batches
        for i in range(len(uni_indices) % num_batches):
            num_uni_indices_in_mega_batch[i] += 1
        
        mega_batches = []
        cur_uni_index = 0
        cur_mm_index = 0
        for i, num_uni_indices in enumerate(num_uni_indices_in_mega_batch):
            mega_batch = []
            mega_batch.extend(uni_indices[cur_uni_index:cur_uni_index + num_uni_indices])
            cur_uni_index += num_uni_indices
            assert len(mega_batch) < self.mega_batch_size

            if i < num_batches - 1:
                increment = self.mega_batch_size - len(mega_batch)
                mega_batch.extend(
                    mm_indices[cur_mm_index:cur_mm_index + increment]
                )
                cur_mm_index += increment
            else: # last batch
                mega_batch.extend(mm_indices[cur_mm_index:])
                assert len(mega_batch) <= self.mega_batch_size, "Last batch is too big."
            
            mega_batches.append(mega_batch)
        
        mega_batch_indices = torch.randperm(len(mega_batches), generator=self.generator)
        mega_batches = [mega_batches[i] for i in mega_batch_indices]
        indices = [i for mega_batch in mega_batches for i in mega_batch]
        return iter(indices)


class WeightedBatchSampler(Sampler):
    r"""
    Sampler that solves the class imbalance for an IQA dataset with optional custom weights.
    If weights are provided, those weights will be used directly for sampling.
    Place the sample data in every mega batch more evenly to avoid severe repetition in some mega batches.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        dataset: torch.utils.data.Dataset,
        sample_weight_decay: Optional[float] = None,
        generator=None,
    ):
        if dataset is None:
            raise ValueError("`dataset` must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.generator = generator
        self.mega_batch_size = batch_size * world_size
        self.data_class = dataset.data_class
        self.weights = dataset.weights
        self.sample_weight_decay = sample_weight_decay  

        assert len(self.data_class) == len(self.weights), "Data class and weights must have the same length."

    def __len__(self):
        return len(self.data_class) 

    def __iter__(self):
        # Step 1: Generate weighted indices with replacement
        weighted_indices = torch.multinomial(
            self.weights, len(self.weights), replacement=True, generator=self.generator
        )

        # Step 2: Shuffle the weighted indices to introduce randomness
        weighted_indices = weighted_indices[torch.randperm(len(weighted_indices), generator=self.generator)]

        # Step 3: Apply the weight decay after shuffling to avoid sequential decay bias
        if self.sample_weight_decay is not None:
            self.weights[weighted_indices] *= self.sample_weight_decay

        # Step 4: Prepare to distribute unique indices evenly across mega batches
        unique_indices = list(set(weighted_indices.tolist()))
        mega_batches = [[] for _ in range((len(weighted_indices) // self.mega_batch_size) + 1)]

        # Create a dictionary to track the counts of each unique index
        index_counts = {idx: (weighted_indices == idx).sum().item() for idx in unique_indices}

        # Distribute unique indices into mega batches evenly
        batch_idx = 0
        while any(count > 0 for count in index_counts.values()):
            for idx in unique_indices:
                if index_counts[idx] > 0:
                    mega_batches[batch_idx].append(idx)
                    index_counts[idx] -= 1
                    batch_idx = (batch_idx + 1) % len(mega_batches)  # Cycle through mega batches

        # Flatten mega batches into a single list of indices
        indices = [i for mega_batch in mega_batches for i in mega_batch]

        # Step 5: Logging and debug information
        iqa_indices = [i for i in indices if self.data_class[i] != "None"]
        unique_final_indices = list(set(indices))
        sampled_class = [self.data_class[i] for i in indices]
        iqa_unique_indices = list(set(iqa_indices))
        use_data_class = Counter(sampled_class).items()

        rank0_print("Data class sampled in this epoch:", use_data_class)
        rank0_print("Number of unique data used in this epoch:", len(unique_final_indices))

        for batch_idx, mega_batch in enumerate(mega_batches[0:5]):
            rank0_print(f"Mega Batch {batch_idx + 1}: {mega_batch}")
        return iter(indices)
        # length of iter_indices should match __len__ method return value
        # for correctly counting the number of epochs in huggingface trainer.
        # and correctly print the sampled data class distribution in each epoch.


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
        return iter_indices


class TrainerWithWeightedSampler(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        sample_weight_decay = self.args.sample_weight_decay
        iter_indices = WeightedBatchSampler(
            self.args.train_batch_size,
            world_size=self.args.world_size * self.args.gradient_accumulation_steps,
            dataset=self.train_dataset,
            sample_weight_decay=sample_weight_decay,
        )
        # rank0_print(list(iter_indices))
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
        # rank0_print(list(iter_indices))
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

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
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
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
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
