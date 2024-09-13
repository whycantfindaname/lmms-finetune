#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=56
#SBATCH --partition=gpu
#SBATCH --exclude=gpu19,gpu3,gpu8,gpu14,gpu17
#SBATCH --job-name=lmms-finetune

eval "$(conda shell.bash hook)"
conda activate lmms-finetune
NUM_GPUS=2
DISTRIBUTED_ARGS="
    --nnodes=1 \
    --nproc_per_node ${NUM_GPUS} \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:0
"

# arguments that are very likely to be changed
# according to your own case
MODEL_ID=qwen-vl-chat # model id; pick on by running `python supported_models.py`
MODEL_PATH=./models/Qwen-VL-Chat   # model path; if None, will download the model from huggingface
CKPT_PATH=checkpoints/qwen-vl-chat_lora-True_qlora-False-no-bbox-8k-vqa/checkpoint-1260                # use --resume_from_checkpoint $CKPT_PATH to continue training from model checkpoint
TRAIN_DATA_PATH=../datasets/train_json # path to the training data json file
EVAL_DATA_PATH=../datasets/val_json # path to the evaluation data json file (optional)
IMAGE_FOLDER=../datasets/images             # path to the image root folder; if provided, the image paths in the json should be relative

TRAIN_VISION_ENCODER=False                              # whether train the vision encoder
USE_VISION_LORA=False                                   # whether use lora for vision encoder (only effective when `TRAIN_VISION_ENCODER` is True)
TRAIN_VISION_PROJECTOR=True                            # whether train the vision projector (only full finetuning is supported)
IQA_DATA=qwen_with_bbox

USE_WEIGHTED_SAMPLE=True

USE_LORA=True                                           # whether use lora for llm
Q_LORA=False                                             # whether use q-lora for llm; only effective when `USE_LORA` is True
LORA_R=64                                                # the lora rank (both llm and vision encoder)
LORA_ALPHA=16                                           # the lora alpha (both llm and vision encoder)

RUN_ID=${MODEL_ID}_lora-${USE_LORA}_qlora-${Q_LORA}-bbox-8k-vqa     # a custom run id that determines the checkpoint folder and wandb run name

DS_STAGE=zero3                                          # deepspeed stage; < zero2 | zero3 >
PER_DEVICE_BATCH_SIZE=1                                 # batch size per GPU
GRAD_ACCUM=1                                            # gradient accumulation steps
NUM_EPOCHS=20                                            # number of training epochs

LR=5e-4                                                 # learning rate
MODEL_MAX_LEN=2048                                       # maximum input length of the model


torchrun $DISTRIBUTED_ARGS train_mix.py \
    --model_id $MODEL_ID \
    --model_path $MODEL_PATH \
    --data_path $TRAIN_DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --output_dir ./checkpoints/$RUN_ID \
    --run_name $RUN_ID \
    --deepspeed ./ds_configs/${DS_STAGE}.json \
    --bf16 True \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate ${LR} \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --weight_decay 0. \
    --warmup_steps 100 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LEN \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --train_vision_encoder $TRAIN_VISION_ENCODER \
    --use_vision_lora $USE_VISION_LORA \
    --train_vision_projector $TRAIN_VISION_PROJECTOR \
    --use_lora $USE_LORA \
    --q_lora $Q_LORA \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --use_weighted_sample $USE_WEIGHTED_SAMPLE \
    --iqa_data $IQA_DATA \
    --resume_from_checkpoint $CKPT_PATH
    