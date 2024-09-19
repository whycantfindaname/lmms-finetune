#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=27
#SBATCH --partition=gpu
#SBATCH --exclude=gpu19,gpu3,gpu8,gpu14,gpu17
#SBATCH --job-name=lmms-finetune
#SBATCH --output=/home/u9820200070/lmms-finetune/output/%j.no_bbox_out.txt
#SBATCH --error=/home/u9820200070/lmms-finetune/output/%j.no_bbox_err.txt

eval "$(conda shell.bash hook)"
conda activate lmms-finetune
NUM_GPUS=4
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

TRAIN_DATA_PATH=datasets/train_json # path to the training data json file
EVAL_DATA_PATH=datasets/val_json # path to the evaluation data json file (optional)
IMAGE_FOLDER=datasets/images             # path to the image root folder; if provided, the image paths in the json should be relative

TRAIN_VISION_ENCODER=True                              # whether train the vision encoder
USE_VISION_LORA=False                                   # whether use lora for vision encoder (only effective when `TRAIN_VISION_ENCODER` is True)
TRAIN_VISION_PROJECTOR=True                            # whether train the vision projector (only full finetuning is supported)
IQA_DATA=koniq_spaq_kadid                          # path to the iqa data json file (optional)

USE_WEIGHTED_SAMPLE=False
SAMPLE_WEIGHT_DECAY=0.7

USE_LORA=False                                           # whether use lora for llm
Q_LORA=False                                             # whether use q-lora for llm; only effective when `USE_LORA` is True
LORA_R=64                                                # the lora rank (both llm and vision encoder)
LORA_ALPHA=16                                           # the lora alpha (both llm and vision encoder)

RUN_ID=${MODEL_ID}_lora-${USE_LORA}_qlora-${Q_LORA}-stage1     # a custom run id that determines the checkpoint folder and wandb run name

DS_STAGE=zero3                                          # deepspeed stage; < zero2 | zero3 >
PER_DEVICE_BATCH_SIZE=16                                 # batch size per GPU
GRAD_ACCUM=1                                            # gradient accumulation steps
NUM_EPOCHS=3                                            # number of training epochs

LR=2e-5                                                 # learning rate
MODEL_MAX_LEN=2048                                       # maximum input length of the model

CKPT_PATH=''               
# use --resume_from_checkpoint $CKPT_PATH to continue training from model checkpoint
# your training settings must be the same as the previous run to resume from the same checkpoint
# for example, your batch size, world size, ds_stage must be the same as the previous run

torchrun $DISTRIBUTED_ARGS train_mix.py \
    --model_id $MODEL_ID \
    --model_path $MODEL_PATH \
    --data_path $TRAIN_DATA_PATH \
    --evaluation_strategy "no" \
    --image_folder $IMAGE_FOLDER \
    --output_dir ./checkpoints/$RUN_ID \
    --report_to wandb \
    --run_name $RUN_ID \
    --deepspeed ./ds_configs/${DS_STAGE}.json \
    --bf16 True \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --eval_strategy "epoch" \
    --save_strategy "steps" \
    --save_steps 188 \
    --save_total_limit 3 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
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
    --sample_weight_decay $SAMPLE_WEIGHT_DECAY 
    