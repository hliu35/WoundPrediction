import numpy as np
import pandas as pd

###########################
#
#  How-to:
#  Set two thresholds: first one would set 1 to everything above that
#                       second threshold would set 0.5 to the two candidates that are above that
#  You will see how many labels are left uncertained.
#
#############################



def thresholding(labels, thresh_1=0.75, thresh_2=0.4):
    uncertained = 0

    for i, row in labels.iterrows():
        if np.count_nonzero(row >= thresh_1) == 1:
            row = np.where(row >= thresh_1, 1, 0)
            labels.iloc[i, :] = row
            continue

        if np.count_nonzero(row >= thresh_2) == 2:
            row = np.where(row >= thresh_2, 0.5, 0.5)
            labels.iloc[i, :] = row
            continue

        else:
            #print(row) 
            uncertained += 1
    
    print("\n\ntotal data:", i)
    print("uncertained data:", uncertained)
    return labels



if __name__ == "__main__":
    file = "../data/labels.csv"
    DF = pd.read_csv(file)

    sigma = np.zeros((4,))

    data = DF.iloc[:, 1:]
    print("sum:\n", np.sum(data, axis=0))

    new_data = thresholding(data, 0.5)
    print("new sum:\n", np.sum(new_data, axis=0))
    


