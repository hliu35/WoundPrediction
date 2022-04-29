import os, sys
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance

# Thresholding parameters
THRESHOLD_1 = 0.75
THRESHOLD_2 = 0.4

# Augmentation parameters
augmenting_probabilities = [0.8, 0.75, 0, 0.4]


def augment(image):
    new_image = image.copy()
    degrees = [0, 90, 180, 270]

    mode = np.random.randint(low=0, high=3)
    if mode == 0: # rotate
        deg = np.random.choice(degrees)
        return new_image.rotate(deg)
    elif mode == 1: # flip
        deg = np.random.choice(degrees)
        new_image = new_image.rotate(deg)
        return ImageOps.flip(new_image)
    elif mode == 2: # brightness
        enhancer = ImageEnhance.Brightness(new_image)
        return enhancer.enhance(factor=np.random.uniform(low=0.5, high=1.5))
    else:
        raise NotImplementedError("Augmentation Mode Not Available")


def thresholding(label, thresh_1, thresh_2):
    modified = True

    if np.count_nonzero(label >= thresh_1) == 1:
        new_label = np.where(label >= thresh_1, 1, 0)
        return modified, new_label

    elif np.count_nonzero(label >= thresh_2) == 2:
        new_label = np.where(label >= thresh_2, 0.5, 0)
        return modified, new_label
    
    return not modified, label


def thresholding_main(DF, thresh_1=THRESHOLD_1, thresh_2=THRESHOLD_2):
    """  
    How-to:
    Set two thresholds: first one would set 1 to everything above that
                        second threshold would set 0.5 to the two candidates that are above that
    You will see how many labels are left uncertained.
    """
    uncertained = 0

    for i, row in DF.iterrows():
        label = row.iloc[1:]
        modified, new_label = thresholding(label, thresh_1, thresh_2)
        DF.iloc[i, 1:] = new_label
        if not modified:
            #print(label)
            uncertained += 1
    
    print("\ntotal data:", i)
    print("uncertained data:", uncertained)

    return DF


def label_stats(dataframe, description):
    print('\n%s'%description)
    print(np.sum(dataframe.iloc[:, 1:], axis=0))


def augment_main(dataframe, datapath = "../data/", outpath="../data_augmented/", probability=0.8):
    '''
    please feed in a dataframe after thresholding
    '''
    # create output folder
    if os.path.exists(outpath):
        shutil.rmtree(outpath)
    os.mkdir(outpath)

    outfile = os.path.join(outpath, "augmented_labels.csv")

    #DF = pd.read_csv("../data/labels.csv")
    #new_DF = thresholding_main(DF, 0.5)
    #print("new sum after thresholding:\n", np.sum(new_DF.iloc[:, 1:], axis=0))

    i, count = 0, 0

    for i, row in dataframe.iterrows():
        # image file
        filename = row.iloc[0]
        filepath = os.path.join(datapath, filename)
        
        # labels
        label = row.iloc[1:]
        stage = np.argmax(label)

        # calculate augmenting probability
        probability = augmenting_probabilities[stage]

        if probability > 0 and np.random.random() <= probability:
            # read old image
            image = Image.open(filepath)
            new_image = augment(image)

            # save image
            new_filename = "aug_"+filename
            new_image.save(os.path.join(outpath, new_filename))

            # save label by appending to old df
            new_row = row.copy()
            new_row["Image"] = new_filename
            dataframe = dataframe.append(new_row)
            count += 1

        # copy original image regardless of augmenting or not
        shutil.copyfile(src=filepath, dst=os.path.join(outpath, filename))     
        
    print("total image:", i+count+1)
    dataframe.to_csv(outfile, index=False)
    return dataframe

    



if __name__ == "__main__":
    # original
    file = "../data/labels.csv"
    DF = pd.read_csv(file)
    label_stats(DF, "original")

    # after thresholding
    new_DF = thresholding_main(DF, 0.5)
    label_stats(new_DF, "after thresholding")

    # after augmenting
    file = "../data_augmented/augmented_labels.csv"
    aug_DF = augment_main(new_DF)
    label_stats(aug_DF, "after augmentation")
