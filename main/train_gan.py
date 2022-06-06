from cProfile import run
import os
import sys
import shutil
import argparse
import random
from itertools import combinations

sys.path.insert(0, '../model')
sys.path.insert(0, '../utils')

import numpy as np

import torchvision.transforms as T
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch
import torch.nn.functional as F



import gan, dcgan
import temporal_encoder as TE
import temporal_classifier as TC

#from dataloader import WoundImageDataset
from dataloader import WoundImagePairsDataset # new dataset with day i and j
from synth_labels import synthesize_softmax_labels as synth_softmax
from synth_labels import synthesize_onehot_labels as synth_onehot



parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches") # changed from 16 to 8
parser.add_argument("--lr", type=float, default=0.0003, help="adam: learning rate") # changed from 0.0002
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space") # original: 100, new: 16
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")  # changed from 64 to 128
parser.add_argument('--n_classes', type=int, default=16, help='number of classes for dataset')
#parser.add_argument('--n_classes', type=int, default=16, help='number of classes for dataset')
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=250, help="interval betwen image samples")
opt = parser.parse_args()


IMG_SHAPE = (opt.channels, opt.img_size, opt.img_size)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"








def train_gan(datapath, annotation_file, outpath="../tmp/"):
    ''' This is the script to train Vanilla GAN '''
    ''' Some lines may be deprecated and need updates '''
    # create output folder
    if os.path.exists(outpath):
        shutil.rmtree(outpath)
    os.mkdir(outpath)

    # retrieve the list of image paths
    img_list = list_full_paths(datapath)
    #print(img_list)
    #exit()

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = gan.Generator(IMG_SHAPE, opt.latent_dim)
    discriminator = gan.Discriminator(IMG_SHAPE)

    if CUDA:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Configure data loader and compose transform functions
    TRANSFORMS = T.Compose([T.ToTensor(), T.Resize((opt.img_size, opt.img_size))])
    #TRANSFORMS = T.Compose([T.Grayscale(num_output_channels=1), T.ToTensor(), T.Resize((opt.img_size, opt.img_size))])
    dataset = WoundImageDataset(img_list, \
        annotation_file, \
        transform = TRANSFORMS) # can also customize transform

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    for epoch in range(opt.epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # SKIP BATCH SIZE OF 1
            if imgs.shape[0] < opt.batch_size: continue

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 100, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], os.path.join(outpath, "%d.png" % batches_done), nrow=4, normalize=True)


