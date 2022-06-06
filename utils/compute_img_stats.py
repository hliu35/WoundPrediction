import numpy as np
import os, sys
from PIL import Image


if __name__ == "__main__":

    filelist = os.listdir("../data_cropped")
    filelist = [os.path.join("../data_cropped/", x) for x in filelist if "png" in x]

    mean = np.zeros((3,))
    sqsum = np.zeros((3,))
    std = np.zeros((3,))
    
    for f in filelist:
        img = np.array(Image.open(f)) / 255.0
        for c in range(3):
            mean[c] += (np.mean(img[:, :, c]) / len(filelist))
            sqsum[c] += np.mean((np.square(img))[:, :, c]) / len(filelist)

    var = sqsum - mean ** 2
    std = np.sqrt(var)

    print(mean, std)