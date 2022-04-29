import os, sys

import numpy as np
import pandas as pd
import PIL as P


def thresholding(label, thresh_1=0.75, thresh_2=0.4):

    if np.count_nonzero(label >= thresh_1) == 1:
        new_label = np.where(label >= thresh_1, 1, 0)
        return new_label

    elif np.count_nonzero(label >= thresh_2) == 2:
        new_label = np.where(label >= thresh_2, 0.5, 0)
        return new_label
    
    return label



def stats_for_thresholding(labels, thresh_1=0.75, thresh_2=0.4):
    ###########################
    #
    #  How-to:
    #  Set two thresholds: first one would set 1 to everything above that
    #                       second threshold would set 0.5 to the two candidates that are above that
    #  You will see how many labels are left uncertained.
    #
    #############################
    uncertained = 0

    for i, row in labels.iterrows():
        label = row.iloc[1:]
        new_label = 
        print(row) 
        uncertained += 1
    
    print("\n\ntotal data:", i)
    print("uncertained data:", uncertained)
    return labels



def label_stats():
    file = "../data/labels.csv"
    DF = pd.read_csv(file)

    sigma = np.zeros((4,))

    data = DF.iloc[:, 1:]
    print("sum:\n", np.sum(data, axis=0))

    new_data = thresholding(data, 0.5)
    print("new sum:\n", np.sum(new_data, axis=0))


def augment():
    image_list = [f for f in os.listdir("../data/") if ".csv" not in f]
    print(image_list)

    annotations = pd.read_csv("../data/labels.csv")

    label_row = self.img_labels.loc[self.img_labels["Image"]==filename]



if __name__ == "__main__":
    pass

