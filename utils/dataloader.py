import os

import numpy as np
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize
from torchvision.io import read_image


def shift(normalized_imgs):
    return normalized_imgs / 2 + 0.5

def unshift(sigmoid_output):
    return (sigmoid_output - 0.5) * 2



def normalize(cropped_img):
    img = np.array(cropped_img).astype(float)
    mask = np.any(img != [0, 0, 0], axis=-1)

    for c in range(3):
        center = img[mask, c].copy()

        mu = np.mean(center)
        sigma = np.std(center)

        img[:, :, c][mask] -= mu
        img[:, :, c][mask] /= sigma

        new_min = np.min(img[mask, c])
        new_max = np.max(img[mask, c])

        img[:, :, c][mask] -= new_min
        img[:, :, c][mask] /= (new_max-new_min)
        img[:, :, c][mask] *= 255

    return img.astype(np.uint8)



class WoundImagePairsDataset(Dataset):
    def __init__(self, combs, annotations_file, transform=None, target_transform=None):
        self.img_list = combs
        self.img_labels = pd.read_csv(annotations_file)

        self.transform = transform 
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        comb = self.img_list[idx]
        filename = comb[2].split("/")[-1]

        data = self.img_labels.loc[self.img_labels["Image"]==filename]
        #image = read_image(img_path) # using pytorch read image yields error in transform ToTensor()
        image_i = Image.open(comb[0])
        image_j = Image.open(comb[1])
        image_k = Image.open(comb[2])

        image_i = normalize(image_i)
        image_j = normalize(image_j)
        image_k = normalize(image_k)

        #Print images for testing
        #image_i.show()
        #image_j.show()
        #image_k.show()

        if self.transform:
            image_i = self.transform(image_i)
            image_j = self.transform(image_j)
            image_k = self.transform(image_k)

        image_i = image_i[:3, :] # rgb no a
        image_j = image_j[:3, :]
        image_k = image_k[:3, :]

        # these are data from day k only
        label = np.fromstring(data.iloc[0, 1][1:-1], dtype=np.float, sep=" ")
        embed = np.fromstring(data.iloc[0, 2][1:-1], dtype=np.float, sep=" ")

        label = torch.FloatTensor(label)
        embed = torch.FloatTensor(embed)

        return (image_i, image_j, image_k, label, embed)
        #return (shift(image_i), shift(image_j), shift(image_k), label, embed)
    
