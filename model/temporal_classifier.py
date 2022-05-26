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


