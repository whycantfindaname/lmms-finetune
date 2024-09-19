import os

# os.environ["WANDB_PROJECT"] = "lmms-ft"
from dataclasses import asdict
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import debugpy
import torch
import transformers
import yaml
from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import deepspeed

from arguments import DataArguments, LoraArguments, ModelArguments, TrainingArguments
from collators import COLLATORS
from mix_datasets import HybridDataset, ValDataset
from IQAdataset import IQADataset
from loaders import LOADERS
from supported_models import MODULE_KEYWORDS
from utils import (
    TrainerWithCustomSampler,
    TrainerWithWeightedSampler,
    find_all_linear_names,
    rank0_print,
    safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3
)

# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception:
#     pass


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = (
        parser.parse_args_into_dataclasses()
    )

    # dumping arguments
    output_dir = getattr(training_args, "output_dir", None)
    assert output_dir is not None, "output_dir is required"
    args_dir = Path(output_dir) / "arguments"
    args_dir.mkdir(parents=True, exist_ok=True)
    yaml.dump(asdict(model_args), open(args_dir / "model.yaml", "w"))
    yaml.dump(asdict(data_args), open(args_dir / "data.yaml", "w"))
    yaml.dump(asdict(training_args), open(args_dir / "training.yaml", "w"))
    yaml.dump(asdict(lora_args), open(args_dir / "lora.yaml", "w"))

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    if getattr(training_args, "deepspeed", None) and getattr(
        lora_args, "q_lora", False
    ):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    device_map = None
    if lora_args.q_lora:
        device_map = (
            {"": int(os.environ.get("LOCAL_RANK") or 0)}
            if int(os.environ.get("WORLD_SIZE", 1)) != 1
            else None
        )
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            raise ValueError("FSDP or ZeRO3 are not incompatible with QLoRA.")

    # llm quantization config (for q-lora)
    bnb_config = None
    if lora_args.use_lora and lora_args.q_lora:
        from transformers import BitsAndBytesConfig

        rank0_print("Quantization for LLM enabled...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
        )

    # load model, tokenizer, processor
    rank0_print("Loading model, tokenizer, processor...")
    if model_args.model_path is None:
        loader = LOADERS[model_args.model_family_id](
            model_path=model_args.model_name_or_path,
            compute_dtype=compute_dtype,
            bnb_config=bnb_config,
            use_flash_attn=training_args.use_flash_attn,
            device_map=device_map,
        )
        # print(model_args.model_name_or_path)
    else:
        loader = LOADERS[model_args.model_family_id](
            model_path=model_args.model_path,
            compute_dtype=compute_dtype,
            bnb_config=bnb_config,
            use_flash_attn=training_args.use_flash_attn,
            device_map=device_map,
        )
    # print(loader)
    model, tokenizer, processor = loader.load()
    tokenizer.model_max_length = training_args.model_max_length

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # freeze certain params
    vision_encoder_keys = MODULE_KEYWORDS[model_args.model_family_id]["vision_encoder"]
    if not training_args.train_vision_encoder:
        rank0_print("Vision encoder is freezed... including:")
        for module in vision_encoder_keys:
            rank0_print(f"\t{module}")
            eval(f"model.{module}").requires_grad_(False)

    vision_projector_keys = MODULE_KEYWORDS[model_args.model_family_id][
        "vision_projector"
    ]
    if not training_args.train_vision_projector:
        rank0_print("Vision projector is freezed... including:")
        for module in vision_projector_keys:
            rank0_print(f"\t{module}")
            eval(f"model.{module}").requires_grad_(False)

    # other components preparation (e.g., image_newline, vision_resampler)
    # we will just freeze these
    if "others" in MODULE_KEYWORDS[model_args.model_family_id]:
        rank0_print("Other multimodal component is freezed... including:")
        for other_key in MODULE_KEYWORDS[model_args.model_family_id]["others"]:
            rank0_print(f"\t{other_key}")
            eval(f"model.{other_key}").requires_grad_(False)

    # lora preparation
    llm_keys = MODULE_KEYWORDS[model_args.model_family_id]["llm"]
    if not (
        lora_args.use_lora
        or (training_args.train_vision_encoder and lora_args.use_vision_lora)
    ):
        rank0_print("No LoRA enabled...")
    else:
        named_modules = {n: m for n, m in model.named_modules()}
        lora_modules = []
        full_modules = []

        if training_args.train_vision_encoder and lora_args.use_vision_lora:
            rank0_print("LoRA for vision encoder enabled...")
            lora_modules.extend(
                find_all_linear_names(named_modules, vision_encoder_keys)
            )
            model.config.freeze_vision_encoder = False
            model.config.use_vision_lora = True
        elif training_args.train_vision_encoder:
            rank0_print("Vision encoder will be fully trained...")
            full_modules.extend(vision_encoder_keys)
            model.config.freeze_vision_encoder = False
            model.config.use_vision_lora = False

        if lora_args.use_lora:
            rank0_print("LoRA for LLM enabled...")
            lora_modules.extend(find_all_linear_names(named_modules, llm_keys))
        else:
            rank0_print("LLM will be fully trained...")
            full_modules.extend(llm_keys)

        if training_args.train_vision_projector:
            rank0_print("Vision projector will be fully trained...")
            full_modules.extend(vision_projector_keys)

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_modules,
            modules_to_save=full_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        # model.config.visual_abstractor_lr = training_args.visual_abstractor_lr

    # print trainable parameters for inspection
    rank0_print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(f"\t{name}")
    model.print_trainable_parameters()

    # load data
    if training_args.use_weighted_sample:
        train_weights = {"bad": 10, "poor": 1, "fair": 8, "good": 50, "excellent": 1000, "None": 0.05}
    else:
        train_weights = None
    train_weights = None
    rank0_print("Loading data...")
    train_iqa_data = data_args.iqa_data + "_train"
    train_dataset = IQADataset(
        data_path=data_args.data_path,
        iqa_data=train_iqa_data,
        image_folder=data_args.image_folder,
        model_family_id=model_args.model_family_id,
        weights=train_weights,
    )

    rank0_print("Length of train dataset:", len(train_dataset))
    rank0_print("train data class:", train_dataset.class_num)
    if data_args.eval_data_path:
        val_iqa_data = data_args.iqa_data + "_val"
        eval_dataset = ValDataset(
            data_path=data_args.eval_data_path,
            iqa_data=val_iqa_data,
            image_folder=data_args.image_folder,
            model_family_id=model_args.model_family_id,
            user_key=data_args.user_key,
            assistant_key=data_args.assistant_key,
        )
        rank0_print("Length of eval dataset:", len(eval_dataset))
        rank0_print("eval data class:", eval_dataset.class_num)
    else:
        eval_dataset = None
        training_args.eval_strategy = "no"

    # data collator
    data_collator = COLLATORS[model_args.model_family_id](
        tokenizer=tokenizer,
        processor=processor,
    )

    # trainer
    rank0_print(f"Use weighted sampler:{training_args.use_weighted_sample}")
    rank0_print("train_dataset class weights:", train_weights)
    if training_args.use_weighted_sample:
        trainer = TrainerWithWeightedSampler(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
    else:
        trainer = TrainerWithCustomSampler(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()

    if lora_args.use_lora:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), lora_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(output_dir)
            model.save_pretrained(output_dir, state_dict=state_dict)
            torch.save(
                non_lora_state_dict,
                os.path.join(output_dir, "non_lora_trainables.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(
            trainer=trainer,
            output_dir=training_args.output_dir,
        )



if __name__ == "__main__":
    train()
