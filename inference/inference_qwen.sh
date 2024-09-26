#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate lmms-finetune
srun python inference/inference_qwen_bbox.py