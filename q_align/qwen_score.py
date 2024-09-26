import json
import os

from scorer import QwenScorer

# setup_debugger()
file = "../datasets/val_json/qwen_with_bbox_val.json"
bbox_file = 'results/detect/output.json'
# 读取JSON文件
with open(file, "r", encoding="utf-8") as file:
    data = json.load(file)
with open(bbox_file, "r", encoding="utf-8") as file:
    bbox_data = json.load(file)
img_list = []
bbox_list = []
image_dir = "../datasets/images"
save_path = "results/mos/qwen/qwen_score_detect_bbox.json"
text = "The distortions present in the image and their locations are as follows: "
os.makedirs(os.path.dirname(save_path), exist_ok=True)
for i in range(len(data)):
    image = data[i]["image"]
    if os.path.dirname(image) != "QualityLLM_single_2w":
        continue
    img_list.append(os.path.join(image_dir, image))
    for item in bbox_data:
        if item["image"] == os.path.basename(image):
            bbox = text +item["bbox"]
    bbox_list.append(bbox)
print(bbox_list[0])
model_base = "models/Qwen-VL-Chat"
model_path = "checkpoints/qwen-vl-chat_lora-True_qlora-False-bbox-8k-sample2"
model_name = os.path.basename(model_path)
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
device = "cuda:0"
scorer = QwenScorer(model_path, model_base, device, levels, model_name)

output = scorer(img_list, bbox_list=bbox_list)
# output = scorer(img_list)
print("Saving results to", save_path)
with open(save_path, "w") as file:
    json.dump(output, file, ensure_ascii=False, indent=4)
