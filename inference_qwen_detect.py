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


# Load the JSON data
image_file = (
    "/home/liaowenjie/桌面/画质大模型/mmdetection/data/iqa/annotations/val.json"
)
images = []
with open(image_file, "r", encoding="utf-8") as file:
    image = json.load(file)["images"]
for item in image:
    images.append(item["file_name"])

json_file1 = "./dataset/json/val_data_bbox.json"
with open(json_file1, "r", encoding="utf-8") as file1:
    train = json.load(file1)

json_file2 = "./dataset/json/train_data_bbox.json"
with open(json_file2, "r", encoding="utf-8") as file2:
    val = json.load(file2)

data = train + val

image_folder = "./dataset/images"

new_json_data = []

for image in images:
    image_path = os.path.join(image_folder, image)
    for i in range(len(data)):
        if data[i]["image"] == image:
            entry = data[i]
    assess_question = entry["conversations"][2]["value"]
    bbox_question = entry["conversations"][4]["value"]
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


with open("qwen_answer_train.json", "w", encoding="utf-8") as file:
    json.dump(new_json_data, file, ensure_ascii=False, indent=4)

print("New JSON file created with generated answers.")
