import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(1234)


# setup_debugger()

model_path = "checkpoints/qwen-vl-chat_lora-True_qlora-False-stage2-no-sample"
model_base = "models/qwenvl_iqa_stage1_full"
tokenizer = AutoTokenizer.from_pretrained(model_base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
).eval().to(torch.float16).to("cuda:1")

val_file = "datasets/val_json/qwen_with_bbox_val.json"

# 读取JSON文件
with open(val_file, "r", encoding="utf-8") as file:
    data = json.load(file)

# 图像文件夹路径
image_folder = "../datasets/images"
save_img_path = 'results/vis/qwen-vl_stage2_no_sample'

qwen_answer = []
kwargs = {"max_new_tokens": 2048, "do_sample": True} 

score_question = "Can you evaluate the quality of the image in a single sentence?"
detailed_question = "Can you provide a detailed evaluation of the image’s quality?"
dist_identification = (
    "What types of distortions are present in the image? Answer with names only."
)
dist_location = "Can you give the bounding box coordinates for the identified distortions in the image?"

for entry in tqdm(data[0:10], total=len(data[0:10])):
    image_name = entry["image"]
    chat = {"image": image_name, "conversation": []}
    image_path = os.path.join(image_folder, image_name)
    save_path = os.path.join(save_img_path, image_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
    query = tokenizer.from_list_format([
    {'image': image_path},
    {'text': response}])
    print(response)
# inputs = tokenizer(query, return_tensors='pt')
    try:
        image = tokenizer.draw_bbox_on_latest_picture(query)
    except Exception as e:
        image = None
        print(e)

    if image:
        image.save(save_path)

    print(history)
    qwen_answer.append(history)

    with open("results/pred/qwen-vl_stage2_no_sample_val_pred.json", "w", encoding="utf-8") as file:
        json.dump(qwen_answer, file, ensure_ascii=False, indent=4)

print("New JSON file created with generated answers.")
