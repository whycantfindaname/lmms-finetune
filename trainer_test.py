from transformers.trainer import *
import math
from collections import Counter
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import transformers
from deepspeed import zero
from torch.utils.data import Sampler
from transformers import Trainer
from transformers.trainer import _is_peft_model

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
        return 2
    # len(self.data_class) 

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
        return iter(indices[0:2])


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
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        iter_indices = WeightedBatchSampler(
            self.args.train_batch_size,
            world_size=self.args.world_size * self.args.gradient_accumulation_steps,
            dataset=self.train_dataset,
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

    def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
        ):
            self.accelerator.free_memory()
            self._train_batch_size = batch_size
            if self.args.auto_find_batch_size:
                if self.state.train_batch_size != self._train_batch_size:
                    from accelerate.utils import release_memory

                    (self.model_wrapped,) = release_memory(self.model_wrapped)
                    self.model_wrapped = self.model

                    # Check for DeepSpeed *after* the intial pass and modify the config
                    if self.is_deepspeed_enabled:
                        # Temporarily unset `self.args.train_batch_size`
                        original_bs = self.args.per_device_train_batch_size
                        self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                        self.propagate_args_to_deepspeed(True)
                        self.args.per_device_train_batch_size = original_bs
                self.state.train_batch_size = self._train_batch_size
            logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
            # Data loader and number of training steps
            train_dataloader = self.get_train_dataloader()  # 有set_epoch方法
            # next(iter(train_dataloader))调用一次train_dataloader,就会调用一次sampler的__init__，__len__,__iter__
            # 返回结果是处理好的模型的inputs: input_ids, attention_mask, labels
            if self.is_fsdp_xla_v2_enabled:
                train_dataloader = tpu_spmd_dataloader(train_dataloader)

            # Setting up training control variables:
            # number of training epochs: num_train_epochs
            # number of training steps per epoch: num_update_steps_per_epoch
            # total number of training steps to execute: max_steps
            total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
            # args.gradient_accumulation_steps还是这次的ga，不是resume ckpt之前的ga
            len_dataloader = None
            num_train_tokens = None
            if has_length(train_dataloader):
                len_dataloader = len(train_dataloader) # sample总数 // self._train_batch_size * world_size
                num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps # sample总数 // self._train_batch_size * world_size * gradient_accumulation_steps
                num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
                num_examples = self.num_examples(train_dataloader)
                if args.max_steps > 0:
                    max_steps = args.max_steps
                    num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                        args.max_steps % num_update_steps_per_epoch > 0
                    )
                    # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                    # the best we can do.
                    num_train_samples = args.max_steps * total_train_batch_size
                    if args.include_tokens_per_second:
                        num_train_tokens = (
                            self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                        )
                else:
                    max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                    num_train_epochs = math.ceil(args.num_train_epochs)
                    num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                    if args.include_tokens_per_second:
                        num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
            elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
                max_steps = args.max_steps
                # Setting a very large number of epochs so we go as many times as necessary over the iterator.
                num_train_epochs = sys.maxsize
                num_update_steps_per_epoch = max_steps
                num_examples = total_train_batch_size * args.max_steps
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
            else:
                raise ValueError(
                    "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                    f" {args.max_steps}"
                )

            if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
                if self.args.n_gpu > 1:
                    # nn.DataParallel(model) replicates the model, creating new variables and module
                    # references registered here no longer work on other gpus, breaking the module
                    raise ValueError(
                        "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                        " (torchrun or torch.distributed.launch (deprecated))."
                    )
                else:
                    debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

            delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

            # We need to reset the scheduler, as its parameters may be different on subsequent calls
            if self._created_lr_scheduler:
                self.lr_scheduler = None
                self._created_lr_scheduler = False

            if self.is_deepspeed_enabled:
                self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

            if not delay_optimizer_creation:
                self.create_optimizer_and_scheduler(num_training_steps=max_steps)
            
            # 改callbacks
            self.state = TrainerState(
                stateful_callbacks=[
                    cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
                ]
            )
            self.state.is_hyper_param_search = trial is not None
            self.state.train_batch_size = self._train_batch_size

            # Compute absolute values for logging, eval, and save if given as ratio
            if args.logging_steps is not None:
                if args.logging_steps < 1:
                    self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
                else:
                    self.state.logging_steps = args.logging_steps
            if args.eval_steps is not None:
                if args.eval_steps < 1:
                    self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
                else:
                    self.state.eval_steps = args.eval_steps
            if args.save_steps is not None:
                if args.save_steps < 1:
                    self.state.save_steps = math.ceil(max_steps * args.save_steps)
                else:
                    self.state.save_steps = args.save_steps

            # Activate gradient checkpointing if needed
            if args.gradient_checkpointing:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

            model = self._wrap_model(self.model_wrapped)
            # 在深度学习训练中，模型可能会被多个模块或包装器（wrappers）层层封装。例如，如果你使用 DeepSpeed 来加速训练，它会先将模型封装成 DeepSpeed 对象。而在分布式训练的场景中，通常还需要将这个 DeepSpeed 封装的模型再进一步包装成 torch.nn.DistributedDataParallel 对象来进行分布式数据并行训练。
            # self.model_wrapped: 指向最外层的模型，即所有封装层之上最终的模型。这是进行前向传播（forward pass）时实际使用的模型。
            # 如果模型没有被封装（例如没有使用 DeepSpeed 或分布式训练），那么 self.model_wrapped 就等于 self.model。
            # 因此，self.model_wrapped 始终指向训练和推理时应该使用的那个模型，即使模型经过了多层封装。通过这种方式，代码可以简化模型的调用过程，而不必担心底层封装的细节。

            # as the model is wrapped, don't use `accelerator.prepare`
            # this is for unhandled cases such as
            # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
            use_accelerator_prepare = True if model is self.model else False

            if delay_optimizer_creation:
                if use_accelerator_prepare:
                    self._fsdp_qlora_plugin_updates()
                    self.model = self.accelerator.prepare(self.model)
                self.create_optimizer_and_scheduler(num_training_steps=max_steps)

            # prepare using `accelerator` prepare
            if use_accelerator_prepare:
                self.model.train()
                if hasattr(self.lr_scheduler, "step"):
                    if self.use_apex:
                        model = self.accelerator.prepare(self.model)
                    else:
                        model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
                        # self.optimizer: AdamW object has two parameter groups
                        # 还有accelerator_state等参数
                        # 可参考Q-Align Trainer
                else:
                    # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                    model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                        self.model, self.optimizer, self.lr_scheduler
                    )
            elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                # In this case we are in DDP + LOMO, which should be supported
                self.optimizer = self.accelerator.prepare(self.optimizer)

            if self.is_fsdp_enabled:
                self.model = self.model_wrapped = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

            # ckpt loading, 前面那个没有load，在这里才load了
            if resume_from_checkpoint is not None:
                if self.is_deepspeed_enabled:
                    deepspeed_load_checkpoint(
                        self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                    )
                elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                    self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

            # Check if saved optimizer or scheduler states exist
            self._load_optimizer_and_scheduler(resume_from_checkpoint)

            # important: at this point:
            # self.model         is the Transformers Model
            # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
            # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

            # Train!
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {num_examples:,}")
            logger.info(f"  Num Epochs = {num_train_epochs:,}")
            logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
            if self.args.per_device_train_batch_size != self._train_batch_size:
                logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
            logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {max_steps:,}")
            logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

            self.state.epoch = 0
            start_time = time.time()
            epochs_trained = 0
            steps_trained_in_current_epoch = 0
            steps_trained_progress_bar = None

            # Check if continuing training from a checkpoint
            if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            ):
                self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
                self.compare_trainer_and_checkpoint_args(self.args, self.state)
                self._load_callback_state()
                epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
                if not args.ignore_data_skip:
                    steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                    steps_trained_in_current_epoch *= args.gradient_accumulation_steps
                else:
                    steps_trained_in_current_epoch = 0

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info(f"  Continuing training from epoch {epochs_trained}")
                logger.info(f"  Continuing training from global step {self.state.global_step}")
                if not args.ignore_data_skip:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch."
                    )

            # Update the references
            self.callback_handler.model = self.model
            self.callback_handler.optimizer = self.optimizer
            self.callback_handler.lr_scheduler = self.lr_scheduler
            self.callback_handler.train_dataloader = train_dataloader
            if self.hp_name is not None and self._trial is not None:
                # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
                # parameter to Train when using DDP.
                self.state.trial_name = self.hp_name(self._trial)
            if trial is not None:
                assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
                self.state.trial_params = hp_params(assignments)
            else:
                self.state.trial_params = None
            # This should be the same if the state has been saved but in case the training arguments changed, it's safer
            # to set this after the load.
            self.state.max_steps = max_steps
            self.state.num_train_epochs = num_train_epochs
            self.state.is_local_process_zero = self.is_local_process_zero()
            self.state.is_world_process_zero = self.is_world_process_zero()

            # tr_loss is a tensor to avoid synchronization of TPUs through .item()
            tr_loss = torch.tensor(0.0).to(args.device)
            # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
            self._total_loss_scalar = 0.0
            self._globalstep_last_logged = self.state.global_step
            model.zero_grad()
            grad_norm: Optional[float] = None
            self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

            if args.eval_on_start:
                self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

            total_batched_samples = 0
            # train了所有epoch
            for epoch in range(epochs_trained, num_train_epochs):
                # 不是torch.utils.data.IterableDataset就不会有set_epoch()方法
                # 但是这里返回的是True
                if hasattr(train_dataloader, "set_epoch"):
                    train_dataloader.set_epoch(epoch)
                # 主要作用是为每个训练 epoch 设置一个新的随机种子
                # 以确保所有进程在同一个 epoch 内的数据加载和随机化操作是同步的
                
                # print(train_dataloader.sampler)  # SequentialSampler, 按照自定义Sampler返回的indices顺序采样
                # Reset the past mems state at the beginning of each epoch if necessary.
                if args.past_index >= 0:
                    self._past = None

                steps_in_epoch = (
                    len(train_dataloader)
                    if len_dataloader is not None
                    else args.max_steps * args.gradient_accumulation_steps
                )
                self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

                if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                    self._load_rng_state(resume_from_checkpoint)

                rng_to_sync = False
                steps_skipped = 0
                if steps_trained_in_current_epoch > 0:
                    train_dataloader = skip_first_batches(train_dataloader, steps_trained_in_current_epoch)
                    steps_skipped = steps_trained_in_current_epoch
                    steps_trained_in_current_epoch = 0
                    rng_to_sync = True

                step = -1
                # 在这里调用了train_dataloader，会print在sample出的样本信息(sampler中定义)
                # 但是只能print出最初的一次
                # 猜测：会不会是在后续的epoch里print看不见了
                # 在下面循环的结果中显示的epoch不是真的epoch，而是step数
                # 如果想重新改变sample顺序，需要重新创建train_dataloader
                # 运行train_dataloader = self.get_train_dataloader()
                for step, inputs in enumerate(train_dataloader):
                    total_batched_samples += 1

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            self.state.num_input_tokens_seen += (
                                torch.sum(
                                    self.accelerator.gather(
                                        torch.tensor(
                                            inputs[main_input_name].numel(), device=self.args.device, dtype=torch.int64
                                        )
                                    )
                                )
                                .cpu()
                                .item()
                            )
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    with self.accelerator.accumulate(model):
                        tr_loss_step = self.training_step(model, inputs)

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss += tr_loss_step

                    # flos检验运算量
                    self.current_flos += float(self.floating_point_ops(inputs))

                    is_last_step_and_steps_less_than_grad_acc = (
                        steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                    )

                    if (
                        total_batched_samples % args.gradient_accumulation_steps == 0
                        or
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        is_last_step_and_steps_less_than_grad_acc
                    ):
                        # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                        # in accelerate. So, explicitly enable sync gradients to True in that case.
                        if is_last_step_and_steps_less_than_grad_acc:
                            self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            # deepspeed does its own clipping

                            if is_sagemaker_mp_enabled() and args.fp16:
                                _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                            if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm

                        self.optimizer.step()

                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                        if optimizer_was_run:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                        self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        # PyTorch/XLA relies on the data loader to insert the mark_step for
                        # each step. Since we are breaking the loop early, we need to manually
                        # insert the mark_step here.
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                if step < 0:
                    logger.warning(
                        "There seems not to be a single sample in your train_dataloader, stopping training at step"
                        f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                        f" num_steps ({max_steps}) higher than the number of available samples."
                    )
                    self.control.should_training_stop = True

                self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
                self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

                if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                    if is_torch_xla_available():
                        # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                        xm.master_print(met.metrics_report())
                    else:
                        logger.warning(
                            "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                            "configured. Check your training configuration if this is unexpected."
                        )
                if self.control.should_training_stop:
                    break
            
                train_dataloader = self.get_train_dataloader()

            if args.past_index and hasattr(self, "_past"):
                # Clean the state at the end of training
                delattr(self, "_past")

            logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
            if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
                # Wait for everyone to get here so we are sure the model has been saved by process 0.
                if is_torch_xla_available():
                    xm.rendezvous("load_best_model_at_end")
                elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                    dist.barrier()
                elif is_sagemaker_mp_enabled():
                    smp.barrier()

                self._load_best_model()

            # add remaining tr_loss
            self._total_loss_scalar += tr_loss.item()
            effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
            train_loss = self._total_loss_scalar / effective_global_step

            metrics = speed_metrics(
                "train",
                start_time,
                num_samples=num_train_samples,
                num_steps=self.state.max_steps,
                num_tokens=num_train_tokens,
            )
            self.store_flos()
            metrics["total_flos"] = self.state.total_flos
            metrics["train_loss"] = train_loss

            self.is_in_train = False

            self._memory_tracker.stop_and_update_metrics(metrics)

            self.log(metrics)

            run_dir = self._get_output_dir(trial)
            checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

            # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
            if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
                for checkpoint in checkpoints_sorted:
                    if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                        shutil.rmtree(checkpoint, ignore_errors=True)

            self.control = self.callback_handler.on_train_end(args, self.state, self.control)

            # Wait for the checkpoint to be uploaded.
            self._finish_current_push()

            # After training we make sure to retrieve back the original forward pass method
            # for the embedding layer by removing the forward post hook.
            if self.neftune_noise_alpha is not None:
                self._deactivate_neftune(self.model)

            return TrainOutput(self.state.global_step, train_loss, metrics)

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
