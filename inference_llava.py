import json
import os

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

device = "cuda:0"
# 模型路径
original_model_id = "./models/llava-interleave-qwen-7b"
model_id = "./checkpoints/llava-interleave-qwen-7b_lora-True_qlora-False-3k"

# 加载模型和处理器
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
).to(device)

processor = AutoProcessor.from_pretrained(original_model_id)
json_file = "./dataset/json/llava_train_gt.json"
# 读取JSON文件
with open(json_file, "r", encoding="utf-8") as file:
    data = json.load(file)

# 图像文件夹路径
image_folder = "./dataset/images"

new_json_data = []

# data1 = [
# item for item in data if item["image"] == "ICME2021_UGC0088_720x1280_30_crf_47.png"
# ]

for entry in data:
    image_name = entry["image"]
    question = entry["conversations"][2]["value"]
    print(question)
    # 构建对话
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image"},
            ],
        },
    ]

    # 应用聊天模板
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # 加载图像
    image_path = os.path.join(image_folder, image_name)
    try:
        raw_image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Image file {image_name} not found in {image_folder}")
        continue

    # 准备输入
    inputs = processor(prompt, raw_image, return_tensors="pt").to(0, torch.float16)

    # 生成输出
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    # print(output)
    # 解码并提取生成的答案
    generated_text = processor.decode(output[0][2:], skip_special_tokens=True).split(
        "\n"
    )[-1]
    print(generated_text)
    # 将结果加入新的JSON数据
    new_json_data.append({"image": image_name, "generated_answer": generated_text})

# 写入新的JSON文件
with open("llava_train_pred.json", "w", encoding="utf-8") as file:
    json.dump(new_json_data, file, ensure_ascii=False, indent=4)

print("New JSON file created with generated answers.")