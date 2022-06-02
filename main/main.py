import numpy as np
import matplotlib.pyplot as plt

import torch



def helloworld(path_i, path_j, desired_k):
    print("Hello World!")

    # This is going to be a wrapper of our model
    
    # The inputs are three things:
    # 1. image path of day i
    # 2. image path of day j
    # 3. which day k the user wants to see

    # we open the images
    # extrapolate
    # put it into GAN
    # and display the output image with matplotlib.pyplot

    # Viola!


if __name__ == "__main__":
    # These are just example prompts
    mice_age = input("Youth or Adult [Y/A]?  ")
    mice_idx = input("Mice index [1~5]?"  )
    i = int(input("Day i?   "))
    j = int(input("Day j?   "))
    k = int(input("What day k do you want to see?   "))

    path_i = ""
    path_j = ""

    helloworld(path_i, path_j, desired_k=k)



