import json
import os

from scorer import QwenQAlignScorer

# setup_debugger()
file = "../gen_prompt/dataset/clean_data_v0/qwen_single/qwen_no_bbox.json"

# 读取JSON文件
with open(file, "r", encoding="utf-8") as file:
    data = json.load(file)
img_list = []
image_dir = "../datasets/images"
save_path = "results/mos/qwen/qwen_with_dist_qalign_score_no_bbox_all.json"
dist_list = []
os.makedirs(os.path.dirname(save_path), exist_ok=True)
for i in range(len(data)):
    image = data[i]["image"]
    img_list.append(os.path.join(image_dir, image))
    dist_list.append((data[i]["conversations"][3]["value"]))
print(dist_list[0])
model_base = "models/Qwen-VL-Chat"
model_path = "checkpoints/qwen-vl-chat_lora-True_qlora-False-no-bbox-8k-vqa/checkpoint-1260"
model_name = "qwen-vl-chat_lora-True_qlora-False-no-bbox-8k-vqa"
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
scorer = QwenQAlignScorer(model_path, model_base, device, levels, model_name)

output = scorer(img_list, dist_list=dist_list)
with open(save_path, "w") as file:
    json.dump(output, file, ensure_ascii=False, indent=4)
