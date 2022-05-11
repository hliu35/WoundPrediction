import os

from PIL import Image

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize

from torchvision.io import read_image

#import augment as AUG

#THRESHOLD_1 = 0.75
#THRESHOLD_2 = 0.4

class WoundImageDataset(Dataset):
    def __init__(self, img_list, annotations_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_list = img_list
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        #return len(self.img_labels)
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        #print(img_path)
        filename = img_path.split("/")[-1]

        data = self.img_labels.loc[self.img_labels["Image"]==filename]
        #image = read_image(img_path) # using pytorch read image yields error in transform ToTensor()
        image = Image.open(img_path)
        #assert(np.all(image[3] == 0))
        #label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)

        image = image[:3, :] # rgb no a
        #label = data.iloc[0, 1:]
        label = np.fromstring(data.iloc[0, 1][1:-1], dtype=np.float, sep=" ")
        embed = np.fromstring(data.iloc[0, 2][1:-1], dtype=np.float, sep=" ")

        #_, label = AUG.thresholding(label,thresh_1=THRESHOLD_1, thresh_2=THRESHOLD_2)    
        label = torch.FloatTensor(label)
        embed = torch.FloatTensor(embed)

        #return image[2], label # use blue channel only
        return image, label, embed
    
