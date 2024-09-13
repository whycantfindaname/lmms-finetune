import json
import os

from scorer import QwenScorer, QwenQAlignScorer

# setup_debugger()
file = "../gen_prompt/dataset/clean_data_v0/qwen_single/qwen_with_bbox.json"

# 读取JSON文件
with open(file, "r", encoding="utf-8") as file:
    data = json.load(file)
img_list = []
bbox_list = []
image_dir = "../datasets/images"
save_path = "results/mos/qwen/qwen_bbox_qalign_score_with_bbox_all.json"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
for i in range(len(data)):
    image = data[i]["image"]
    img_list.append(os.path.join(image_dir, image))
    bbox_list.append(data[i]["conversations"][7]["value"])
print(bbox_list[0])
model_base = "models/Qwen-VL-Chat"
model_path = "checkpoints/qwen-vl-chat_lora-True_qlora-False-bbox-8k-vqa/checkpoint-1260"
model_name = 'qwen-vl-chat_lora-True_qlora-False-bbox-8k-vqa'
levels = [
    " excellent",
    " good",
    " fair",
    " poor",
    " bad",
    " high",
    " low",
    " fine",
    " moderate",
    " decent",
    " average",
    " medium",
    " acceptable",
]
device = "cuda:1"
scorer = QwenQAlignScorer(model_path, model_base, model_name=model_name, device=device, level=levels)

output = scorer(img_list, bbox_list=bbox_list)
with open(save_path, "w") as file:
    json.dump(output, file, ensure_ascii=False, indent=4)
