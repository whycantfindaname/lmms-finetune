# Finetune Qwen-VL for IQA tasks
Finetune Qwen-VL for recognizing distortions in the image, output bbox inforamtion of distorted regions, and predicting the quality of the image(including a level and a score).
## Installation

```python
# clone this repo
git clone -b develop https://github.com/whycantfindaname/lmms-finetune.git

# set up a conda environment
conda create -n lmms-finetune python=3.10 -y
conda activate lmms-finetune
## this will install the latest version of torch
## feel free to change it to a specific version
python -m pip install -r requirements.txt

## optionally install flash attention
python -m pip install --no-cache-dir --no-build-isolation flash-attn

```

## Dataset
run `python download_iqa.py` to download the SPAQ, KONIQ, KADID10K datasets.
The datasets should be organized as follows:
```
├── datasets
│   ├── images
│   │   ├── spaq
│   │   ├── koniq
│   │   ├── kadid10k
│   |   ├── QualityLLM_single_2w
|   ├── train_json
|       ├── koniq_spaq_kadid_train.json
|       ├── qwen_with_bbox_train.json
|   ├── val_json
|       ├── test_kadid.json
|       |── test_koniq.json
|       |── test_spaq.json
|       |── qwen_with_bbox_val.json
```

## Qwen-VL finetuning stage 1
run `bash stage1_full.sh` to fully finetune Qwen-VL on the SPAQ, KONIQ, and KADID10K datasets to enable model quality level and score prediction.

The hyperparameters settings are the same as the original paper [Q-Align](https://github.com/Q-Future/Q-Align).

Better using four A100 GPUs as the original paper did.

After the training, run `merge_lora_weights_and_save_hf_model.py` to merge the LORA weights and save the finetuned model in hub format in local directory "Qwen-VL-IQA-stage1".

## Qwen-VL finetuning stage 2
run `bash stage2_lora.sh` to finetune Qwen-VL on the QualityLLM_single_2w dataset to enable mdoel bbox prediction and detailed image quality assessment. 

After the training, run `merge_lora_weights_and_save_hf_model.py` to merge the LORA weights and save the finetuned model in hub format in local directory "Qwen-VL-IQA-stage2".