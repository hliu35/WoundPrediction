import os

from PIL import Image

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize

from torchvision.io import read_image

THRESHOLD_1 = 0.75
THRESHOLD_2 = 0.4


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
        label = label_row.iloc[0, 1:]

        label = self.thresholding(label,)    
        label = torch.FloatTensor(label).T
        

        return image, label
    

    def thresholding(self, label, thresh_1=THRESHOLD_1, thresh_2=THRESHOLD_2):

        if np.count_nonzero(label >= thresh_1) == 1:
            new_label = np.where(label >= thresh_1, 1, 0)
            return new_label

        elif np.count_nonzero(label >= thresh_2) == 2:
            new_label = np.where(label >= thresh_2, 0.5, 0)
            return new_label
        
        return label
