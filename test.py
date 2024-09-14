from transformers import AutoModelForCausalLM

model_path = "Qwen-VL-IQA"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")

print(model.hf_device_map)

