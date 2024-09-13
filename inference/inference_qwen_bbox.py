import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(1234)


# setup_debugger()

model_path = "checkpoints/qwen-vl-chat_lora-True_qlora-False-bbox-8k-vqa"
model_base = "models/Qwen-VL-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="cuda", trust_remote_code=True, fp16=True
).eval()

val_file = "dataset/val_json/qwen_no_bbox_val.json"

# 读取JSON文件
with open(val_file, "r", encoding="utf-8") as file:
    data = json.load(file)

# 图像文件夹路径
image_folder = "./dataset/images"

qwen_answer = []
kwargs = {"max_new_tokens": 2048, "do_sample": False}  # , "temperature": float(10)}

score_question = "Can you evaluate the quality of the image in a single sentence?"
detailed_question = "Can you provide a detailed evaluation of the image’s quality?"
dist_identification = (
    "What types of distortions are present in the image? Answer with names only."
)
dist_location = "Can you give the bounding box coordinates for the identified distortions in the image?"

for entry in tqdm(data, total=len(data)):
    image_name = entry["image"]
    chat = {"image": image_name, "conversation": []}
    image_path = os.path.join(image_folder, image_name)
    query = tokenizer.from_list_format(
        [
            {"image": image_path},
            {"text": dist_identification,},
        ]
    )
    response, history = model.chat(tokenizer, query=query, history=None, **kwargs)

    response, history = model.chat(
        tokenizer, detailed_question, history=history, **kwargs
    )

    response, history = model.chat(tokenizer, dist_location, history=history, **kwargs)
    print(history)

    qwen_answer.append(history)


with open("results/pred/qwen_with_bbox_val_pred.json", "w", encoding="utf-8") as file:
    json.dump(qwen_answer, file, ensure_ascii=False, indent=4)

print("New JSON file created with generated answers.")
