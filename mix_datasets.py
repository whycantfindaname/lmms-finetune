import json
import os
from collections import Counter
from typing import Dict, List, Optional
import random
import av
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from IQAdataset import IQADataset
from VQAdataset import VQADataset
from utils import rank0_print
from itertools import chain

class HybridDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        model_family_id: str,
        image_folder,
        samples_per_epoch=20000,  # 在每个 epoch 中处理的总样本数量为 20000
        dataset="iqa||vqa",
        iqa_data="qwen_with_bbox_train",
        vqa_data="llava_instruct_150k",
        weights: Optional[dict] = None,
    ):
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.image_folder = image_folder

        self.datasets = dataset.split("||")
        self.all_datasets = []
        self.data_class = []
        self.weights = weights
        for dataset in self.datasets:
            if dataset == "iqa":
                self.iqa_dataset = IQADataset(
                        data_path,
                        model_family_id,
                        iqa_data,
                        image_folder,
                        weights=weights
                    )
                self.all_datasets.append(
                    self.iqa_dataset
                )
                self.data_class.extend(
                    self.iqa_dataset.data_class
                )
                rank0_print(len(self.iqa_dataset))
            elif dataset == "vqa":
                vqa_samples_per_epoch = samples_per_epoch - len(self.iqa_dataset)
                self.vqa_dataset = VQADataset(
                        data_path,
                        model_family_id,
                        vqa_data,
                        image_folder,
                        vqa_samples_per_epoch,
                        weights=weights
                    )
                self.all_datasets.append(    
                    self.vqa_dataset
                )
                self.data_class.extend(
                    self.vqa_dataset.data_class
                )
                rank0_print(len(self.vqa_dataset))

        self.class_num = Counter(self.data_class).items()
        self.merged_dataset = list(chain(*self.all_datasets))
    
    def __len__(self):
        assert len(self.merged_dataset) == self.samples_per_epoch, "Dataset length is not samples_per_epoch"
        return self.samples_per_epoch

    def __getitem__(self, idx):
        try:
            return self.merged_dataset[idx]
        except Exception:
            print(idx)

class ValDataset(IQADataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

