#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=56
#SBATCH --partition=gpu
#SBATCH --exclude=gpu19,gpu3,gpu8,gpu14,gpu17
#SBATCH --job-name=lmms-finetune
#SBATCH --output=/home/u9920230028/lmms-finetune/testbug/finetune_job_output.txt
#SBATCH --error=/home/u9920230028/lmms-finetune/testbug/finetune_job_error.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jasonliaonk21@gmail.com

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
MODEL_ID=llava-interleave-qwen-7b  # model id; pick on by running `python supported_models.py`
MODEL_PATH=./models/llava-interleave-qwen-7b    # model path; if None, will download the model from huggingface
TRAIN_DATA_PATH=./dataset/json/train_data_bbox.json # path to the training data json file
EVAL_DATA_PATH=./dataset/json/val_data_bbox.json   # path to the evaluation data json file (optional)
IMAGE_FOLDER=./dataset/images                  # path to the image root folder; if provided, the image paths in the json should be relative
VIDEO_FOLDER=./example_data/videos                      # path to the video root folder; if provided, the video paths in the json should be relative
NUM_FRAMES=8                                    # if `num_frames` is not specified in dataset entries, this value will be used to sample frames from videos

TRAIN_VISION_ENCODER=False                              # whether train the vision encoder
USE_VISION_LORA=False                                   # whether use lora for vision encoder (only effective when `TRAIN_VISION_ENCODER` is True)
TRAIN_VISION_PROJECTOR=True                            # whether train the vision projector (only full finetuning is supported)

USE_LORA=True                                           # whether use lora for llm
Q_LORA=False                                            # whether use q-lora for llm; only effective when `USE_LORA` is True
LORA_R=64                                                # the lora rank (both llm and vision encoder)
LORA_ALPHA=16                                           # the lora alpha (both llm and vision encoder)

RUN_ID=${MODEL_ID}_lora-${USE_LORA}_qlora-${Q_LORA}-3k     # a custom run id that determines the checkpoint folder and wandb run name

DS_STAGE=zero3                                          # deepspeed stage; < zero2 | zero3 >
PER_DEVICE_BATCH_SIZE=2                                 # batch size per GPU
GRAD_ACCUM=1                                            # gradient accumulation steps
NUM_EPOCHS=10                                            # number of training epochs

LR=2e-5                                                 # learning rate
MODEL_MAX_LEN=2048                                       # maximum input length of the model


torchrun $DISTRIBUTED_ARGS train.py \
    --model_id $MODEL_ID \
    --model_path $MODEL_PATH \
    --data_path $TRAIN_DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --num_frames $NUM_FRAMES \
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
    --save_strategy "epoch" \
    --save_total_limit 1 \
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
    --lora_alpha $LORA_ALPHA
    