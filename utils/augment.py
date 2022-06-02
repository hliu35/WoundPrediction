import os, sys
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance

# Set Random Seed
np.random.seed(10)


# Thresholding parameters
THRESHOLD_1 = 0.75
THRESHOLD_2 = 0.4
STD_MAX = 0.3

# Augmentation parameters
augmenting_probabilities = [0.90, 0.60, 0.0, 0.45]


def from_string(embedding_str):
    embedding_arr = np.fromstring(embedding_str[1:-1], dtype=np.float, sep=",")
    return embedding_arr



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
    
    elif np.std(label) < STD_MAX and np.max(label) >= 0.5: # general case
        new_label = np.zeros(label.shape)
        new_label[np.argmax(label)] = 1.0
        return modified, new_label
    
    else:
        print(np.std(label))
        return not modified, label


def thresholding_main(DF, thresh_1=THRESHOLD_1, thresh_2=THRESHOLD_2):
    """  
    How-to:
    Set two thresholds: first one would set 1 to everything above that
                        second threshold would set 0.5 to the two candidates that are above that
    You will see how many labels are left uncertained.
    """
    uncertained = 0
    print()

    for i, row in DF.iterrows():
        label = row.iloc[1:]
        modified, new_label = thresholding(label, thresh_1, thresh_2)
        DF.iloc[i, 1:] = new_label
        if not modified: # if the label is uncertained
            print(row)
            uncertained += 1
    
    print("\ntotal data:", i+1)
    print("uncertained data:", uncertained)

    return DF


def label_stats(df, description):
    print('\n%s'%description)
    if df.shape[1] > 3:
        print(np.sum(df.iloc[:, 1:5], axis=0))
    else:
        print(np.sum(df.iloc[:, 1], axis=0))


def augment_main(df_labels, df_embeddings, datapath = "../data/", outpath="../data_augmented/", Pconfirm = 0.8):
    '''
    please feed in a df_labels after thresholding
    '''
    # create output folder
    if os.path.exists(outpath):
        shutil.rmtree(outpath)
    os.mkdir(outpath)

    # define output file directory
    outfile = os.path.join(outpath, "augmented_data.csv")

    # rows in out df
    new_images = []
    new_labels = []
    new_embeddings = []

    #DF = pd.read_csv("../data/labels.csv")
    #new_DF = thresholding_main(DF, 0.5)
    #print("new sum after thresholding:\n", np.sum(new_DF.iloc[:, 1:], axis=0))

    i, count = 0, 0

    for i, row in df_labels.iterrows():
        # image file
        filename = row.iloc[0]
        filepath = os.path.join(datapath, filename)
        
        # labels
        label = row.iloc[1:]
        stage = np.argmax(label)

        # embeddings
        emb_row = df_embeddings.loc[df_embeddings['Mouse ID'] == filename]
        embedding = emb_row["Embedding"].to_numpy()[0]

        # calculate augmenting probability P_aug
        Paug = augmenting_probabilities[stage]

        if Paug > 0 and np.random.random() <= (Paug * Pconfirm): # if criteria is satisfied
            # read old image
            image = Image.open(filepath)
            new_image = augment(image)

            # save new image
            new_filename = "aug_"+filename
            new_image.save(os.path.join(outpath, new_filename))

            # save label by appending to out_df
            new_images.append(new_filename)
            new_labels.append(label.to_numpy())
            new_embeddings.append(embedding)

            # increment counter
            count += 1
        
        # save old variables to lists
        new_images.append(filename)
        new_labels.append(label.to_numpy())
        new_embeddings.append(embedding)

        # copy original image regardless of augmenting or not
        shutil.copyfile(src=filepath, dst=os.path.join(outpath, filename))     
    

    print("\ntotal image after augmentation:", i+count+1)

    # put new columns into the out_df
    data = {"Image":new_images, "Label":new_labels, "Embedding":new_embeddings}
    out_df = pd.DataFrame(data)

    out_df.to_csv(outfile, index=False)
    return out_df

    



if __name__ == "__main__":
    # original source files
    label_file = "../data/labels.csv"
    #embedding_file = "../data/embeddings.csv"
    embedding_file = "../data/new_embeddings.csv"
    
    # load DFs
    DF_labels = pd.read_csv(label_file) # load labels
    DF_embeddings = pd.read_csv(embedding_file) # load embeddings
    DF_embeddings["Embedding"] = DF_embeddings["Embedding"].apply(lambda x: from_string(x)) # change str to np.array

    
    # print some stats
    label_stats(DF_labels, "original")


    # after thresholding
    #new_DF = thresholding_main(DF_labels, thresh_1=0.54, thresh_2=0.3) # defaults to global THRESHOLD_1 and THRESHOLD_2
    DF_labels_new = thresholding_main(DF_labels,)
    label_stats(DF_labels_new, "after thresholding")


    # after augmenting
    # label_file = "../data_augmented/augmented_labels.csv"
    #aug_DF = augment_main(new_DF, datapath="../data/", Pconfirm=1.0) # can use ../data/ or ../data_cropped/
    aug_DF = augment_main(DF_labels_new, DF_embeddings, datapath="../data_cropped/", Pconfirm=1.0)
    label_stats(aug_DF, "after augmentation")

    #print(type(aug_DF.iloc[0, 1]))
