import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


CUDA = True if torch.cuda.is_available() else False

# some parameters
N_CHANNELS = 3 # input channels (r, g, b) = 3


class SiameseCNN(nn.Module):
    # initializers
    def __init__(self, img_shape, latent_dim):
        super(SiameseCNN, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding=1, padding_mode="zeros") # outsize: insize + 2*padding - kernel + 1
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding=1, padding_mode="zeros") # outsize: insize + 2*padding - kernel + 1
        #18 * 64 * 64 input features, 512 output features (see sizing flow below)

        self.fc1 = nn.Linear(int(32 * img_shape[1]//2//2 * img_shape[2]//2//2), 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, latent_dim)


    # forward method
    def forward(self, img):

        x = F.leaky_relu(self.conv1(img), 0.2)
        x = self.pool(x)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = self.pool(x)
        x = x.view(x.size(0),-1)
        
        x = F.leaky_relu(self.fc1(x), 0.2) # test w/o batchnorm
        x = F.leaky_relu(self.fc2(x), 0.2) # test w/o batchnorm
        x = self.fc3(x) # test w/o batchnorm
        
        return x




class Classifier(nn.Module):
    def __init__(self, latent_dim):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(latent_dim * 2, latent_dim // 2)
        self.fc2 = nn.Linear(latent_dim // 2, 1)

    def forward(self, u, v):
        z = torch.cat((u, v), dim=1)
        z = F.relu(self.fc1(z))
        logit = self.fc2(z)
        pred = torch.sigmoid(logit)

        return pred



class TemporalClassifier(nn.Module):
    def __init__(self, img_shape, latent_dim=32):
        super(TemporalClassifier, self).__init__()
        self.latent_dim = latent_dim

        self.cnn = SiameseCNN(img_shape, latent_dim)
        self.classifier = Classifier(latent_dim)



    def forward(self, img_1, img_2):
        # img_1 should be a predecessor of img_2
        u = self.cnn(img_1)
        v = self.cnn(img_2)
        pred = self.classifier(u,v)

        return pred

        