from cProfile import run
import os
import sys
import shutil

sys.path.insert(0, '../model')
sys.path.insert(0, '../utils')

import torchvision.transforms as T
from torchvision.utils import save_image

from torch.utils.data import DataLoader

import torch




import dcgan
from train_cgan import list_full_paths

#from dataloader import WoundImageDataset
from dataloader import WoundImagePairsDataset # new dataset with day i and j


CUDA = True
IMG_SHAPE = (3, 128, 128)
# normalization parameters
MEAN = torch.tensor([0.56014212, 0.40342121, 0.32133712])
STD = torch.tensor([0.20345279, 0.14542403, 0.12238597])


def test_cgan(datapath, annotation_file, outpath="../test_results/"):
    ''' This is the script to TEST Conditional GAN a.k.a. DCGAN '''
    ''' It will only generate day 2-15 of mice a8-1-r '''
    B = 14

    # create output folder
    if os.path.exists(outpath):
        shutil.rmtree(outpath)
    os.mkdir(outpath)

    # retrieve the list of image paths
    img_pairs = list_full_paths(datapath, mode="test")
    test_imgs = []

    for i in range(2, 16):
        text = "/Day %d_A8-1-R"%i
        for p in img_pairs:
            if text in p[2]:
                test_imgs.append(p)
                break
    
    for x in test_imgs:
        print(x[2])

    # Initialize Generator and discriminator
    generator = torch.load("../tmp/cgan_gen.pth")

    
    if CUDA:
        generator.cuda()


    # Configure data loaders and compose transform functions
    TRANSFORMS = T.Compose([T.ToTensor(), T.Resize((128, 128))]) # doesn't need transform here because we won't put them into Discriminators

    test_dataset = WoundImagePairsDataset(test_imgs, annotation_file, transform = TRANSFORMS)
    test_dataloader = DataLoader(test_dataset, batch_size=B, shuffle=False)

    for _, (_, _, imgs_k, _, Y16) in enumerate(test_dataloader):
        y_disp = Y16.cuda() # labels for display
        
        # sample noise for generating fakes
        noise = torch.randn((B, 100)).cuda()
        imgs_k = imgs_k.type(torch.FloatTensor).cuda()


        # y_disp was defined at the very beginning of training
        gen_imgs = generator(noise, y_disp).view(-1, *IMG_SHAPE)
        
        save_image(gen_imgs.data, os.path.join(outpath, 'generated_a8-1-r-day2to15.png'), nrow=7, normalize=False) 
        save_image(imgs_k.data, os.path.join(outpath, 'true_a8-1-r-day2to15.png'), nrow=7, normalize=False) # real data
        break

if __name__ == "__main__":
    datapath = "../data_augmented/"
    annotation_file = "../data_augmented/augmented_data.csv"
    test_cgan(datapath, annotation_file)