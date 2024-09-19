import json
import os
from collections import Counter
from typing import Dict, List, Optional
from utils import rank0_print
import av
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

TO_LOAD_IMAGE: Dict[str, bool] = {
    "llava-1.5": True,
    "llava-1.6": True,
    "llava-interleave": True,
    "llava-next-video": True,
    "qwen-vl": False,
    "phi3-v": True,
    "qwen2-vl": True,
}

def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


class IQADataset(Dataset):

    def __init__(
        self,
        data_path: str,
        model_family_id: str,
        iqa_data: str = "qwen_with_bbox_train",
        image_folder: Optional[str] = None,
        user_key: str = "human",
        assistant_key: str = "gpt",
        weights: Optional[dict] = None,
    ) -> None:
        super(IQADataset, self).__init__()
        with open(os.path.join(data_path, "{}.json".format(iqa_data))) as f:
            self.list_data_dict = json.load(f)
        self.image_folder = image_folder

        self.load_image = TO_LOAD_IMAGE[model_family_id]
        self.user_key = user_key
        self.assistant_key = assistant_key

        self.is_text_only = [
            "image" not in source and "video" not in source
            for source in self.list_data_dict
        ]

        # level is among [excellent, good, fair, poor, bad]."
        try:
            self.data_class = [source.get("quality", source.get("level")) for source in self.list_data_dict]
            self.class_num = Counter(self.data_class).items()
        except Exception:
            pass

        rank0_print("iqa_data: ", len(self.list_data_dict))

        # 如果提供了自定义权重，直接使用它们，否则计算默认权重
        if weights is not None:
            if not isinstance(weights, dict):
                raise AttributeError("weights should be dict like {'class': weights}")

            self.weights = self._apply_custom_weights(weights)
        else:
            self.weights = self._calculate_weights()
        
        assert len(self.weights) == len(self.list_data_dict), "Weights length not match"

    def _apply_custom_weights(self, weights: Dict[str, float]):
        """
        Directly apply custom weights to each sample based on the provided weights.
        """
        weights = [weights[cls] for cls in self.data_class]
        return torch.DoubleTensor(weights)

    def _calculate_weights(self):
        """
        Calculate the weights for each sample based on the class frequency.
        This method is used only when custom weights are not provided.
        """

        # 统计每个类别的样本数量
        class_counts = Counter(self.data_class)
        # 计算默认的类别权重 (1 / 类别样本数量)
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        # 为每个样本分配权重
        weights = [class_weights[cls] for cls in self.data_class]
        return torch.DoubleTensor(weights)
   
    def __len__(self) -> int:
        return 1
    # len(self.list_data_dict)

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
