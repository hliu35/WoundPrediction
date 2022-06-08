import numpy as np
import os, sys
from PIL import Image


from circular_crop import create_circular_mask


if __name__ == "__main__":

    filelist = os.listdir("../data_cropped")
    filelist = [os.path.join("../data_cropped/", x) for x in filelist if "png" in x]

    mean = np.zeros((3,))
    sqsum = np.zeros((3,))
    std = np.zeros((3,))

    minimum, maximum = np.zeros((3,)), np.zeros((3,))
    
    for f in filelist:
        img = np.array(Image.open(f)) / 255.0
        #mask = np.any(img != [0, 0, 0], axis=-1)
        mask = create_circular_mask(h=352, w=352)
        for c in range(3):
            mean[c] += (np.mean(img[:, :, c][mask]) / len(filelist))
            sqsum[c] += np.mean(np.square(img[:, :, c][mask])) / len(filelist)
            minimum[c] = min(minimum[c], np.min(img[:, :, c][mask]))
            maximum[c] = max(maximum[c], np.max(img[:, :, c][mask]))

    var = sqsum - mean ** 2
    std = np.sqrt(var)

    print(mean, std)
    print((minimum-mean)/std, (maximum-mean)/std)