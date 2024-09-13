import json
import os
import random

import cv2
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
import json
import os
from collections import Counter
from typing import Dict, List, Optional
from utils import rank0_print
import av
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

TO_LOAD_IMAGE: Dict[str, bool] = {
    "llava-1.5": True,
    "llava-1.6": True,
    "llava-interleave": True,
    "llava-next-video": True,
    "qwen-vl": False,
    "phi3-v": True,
}


class VQADataset(Dataset):

    def __init__(
        self,
        data_path: str,
        model_family_id: str,
        vqa_data: str = "llava_instruct_150k",
        image_folder: Optional[str] = None,
        samples_per_epoch=20000,
        user_key: str = "human",
        assistant_key: str = "gpt",
        weights: Optional[dict] = None,
    ) -> None:
        super(VQADataset, self).__init__()
        self.samples_per_epoch = samples_per_epoch
        with open(os.path.join(data_path, "{}.json".format(vqa_data))) as f:
            self.list_data_dict = json.load(f)
        self.image_folder = image_folder

        self.load_image = TO_LOAD_IMAGE[model_family_id]
        self.user_key = user_key
        self.assistant_key = assistant_key

        self.data_class = ["None"] * len(self.list_data_dict)
        self.class_num = Counter(self.data_class).items()

        self.weights = weights

        rank0_print("vqa_data: ", len(self.list_data_dict))

        self.sample_vqa_dataset(samples_per_epoch)

    def sample_vqa_dataset(self, samples_per_epoch):
        total_vqa_samples = len(self.list_data_dict)
        sampled_indices = random.sample(range(total_vqa_samples), samples_per_epoch)
        self.list_data_dict = [self.list_data_dict[i] for i in sampled_indices]
        self.data_class = [self.data_class[i] for i in sampled_indices]
    
    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, i) -> Dict[str, List]:
        source = self.list_data_dict[i]

        images = []
        if "image" in source:
            # here we do not do any image preprocessing but rather
            # let the processor handle everything
            # in some cases this may cause slight differences
            # but should totally be fine (e.g., official llava-1.5 does padding,
            # but llava-1.5-hf (huggingface's implementation) does not)
            if isinstance(source["image"], list):
                image_sources = source["image"]
            elif isinstance(source["image"], str):
                image_sources = [source["image"]]
            else:
                raise ValueError(f"Invalid image source type: {type(source['image'])}")

            for image_path in image_sources:
                if self.image_folder is not None:
                    image_path = os.path.join(self.image_folder, image_path)
                    assert os.path.exists(image_path), f"{image_path} doesn't exist"
                images.append(
                    Image.open(image_path).convert("RGB")
                    if self.load_image
                    else image_path
                )


        system_prompt = None
        if "system_prompt" in source:
            system_prompt = source["system_prompt"]

        convs = []
        assert len(source["conversations"]) > 0, "No conversations found"
        for i, conv in enumerate(source["conversations"]):
            assert conv["from"] == (
                self.user_key if i % 2 == 0 else self.assistant_key
            ), "Invalid conversation"
            convs.append(conv["value"])
        assert len(convs) % 2 == 0, "Odd number of conversations"

        return dict(
            images=images,
            conversations=convs,
            system_prompt=system_prompt,
        )

