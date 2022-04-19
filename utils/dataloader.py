import os

from PIL import Image

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize

from torchvision.io import read_image


class WoundImageDataset(Dataset):
    def __init__(self, img_list, annotations_file="../data/labels.csv", transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_list = img_list
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        #return len(self.img_labels)
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        filename = img_path.split("/")[-1]

        label_row = self.img_labels.loc[self.img_labels["Image"]==filename]
        #image = read_image(img_path) # using pytorch read image yields error in transform ToTensor()
        image = Image.open(img_path)
        #label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        #if self.target_transform:
        #    label = self.target_transform(label)
        image = image[:3, :]
        label = torch.as_tensor(label_row.iloc[0, 1:]).T

        return image, label
