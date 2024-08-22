import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda:1"
original_model_id = "./models/Qwen-VL-Chat"
model_id = "./checkpoints/qwen-vl-chat_lora-True_qlora-False-3k"

model = (
    AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    .to(device)
    .eval()
)


tokenizer = AutoTokenizer.from_pretrained(original_model_id, trust_remote_code=True)

json_file1 = "./dataset/json/qwen_train_data.json"
with open(json_file1, "r", encoding="utf-8") as file1:
    train = json.load(file1)

json_file2 = "./dataset/json/qwen_val_data.json"
with open(json_file2, "r", encoding="utf-8") as file2:
    val = json.load(file2)

data = train

image_folder = "./dataset/images"

new_json_data = []

for item in data:
    image = item["image"]
    image_path = os.path.join(image_folder, image)

    assess_question = item["conversations"][2]["value"]
    bbox_question = item["conversations"][4]["value"]
    print(assess_question)
    print(bbox_question)
    # 1st dialogue turn
    query = tokenizer.from_list_format(
        [
            {"image": image_path},
            {"text": assess_question},
        ]
    )
    assess, history = model.chat(tokenizer, query=query, history=None)
    print(assess)
    # 2nd dialogue turn
    bbox, history = model.chat(
        tokenizer,
        bbox_question,
        history=history,
    )
    print(bbox)

    new_json_data.append(
        {
            "image": image,
            "assess_question": assess_question,
            "answer": assess,
            "bbox_question": bbox_question,
            "bbox": bbox,
        }
    )


with open("generated_answers_detect.json", "w", encoding="utf-8") as file:
    json.dump(new_json_data, file, ensure_ascii=False, indent=4)

print("New JSON file created with generated answers.")
