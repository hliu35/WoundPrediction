from cProfile import run
import os
import sys
import shutil
import argparse

sys.path.insert(0, '../model')
sys.path.insert(0, '../utils')

import numpy as np
import scipy
import matplotlib.pyplot as plt

import torchvision.transforms as T
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch


import gan, dcgan
from dataloader import WoundImageDataset



parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()


img_shape = (opt.channels, opt.img_size, opt.img_size)
CUDA = True if torch.cuda.is_available() else False



def list_full_paths(directory):
    full_list = [os.path.join(directory, file) for file in os.listdir(directory) if "label" not in file]
    return sorted(full_list)


def run_gan(datapath="../data/", outpath="../tmp/"):

    # create output folder
    if os.path.exists(outpath):
        shutil.rmtree(outpath)
    os.mkdir(outpath)

    # retrieve the list of image paths
    img_list = list_full_paths(datapath)
    print(img_list[:10])

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = gan.Generator(img_shape, opt.latent_dim)
    discriminator = gan.Discriminator(img_shape)

    if CUDA:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Configure data loader and compose transform functions
    TRANSFORMS = T.Compose([T.ToTensor(), T.Resize((opt.img_size, opt.img_size))])
    dataset = WoundImageDataset(img_list, transform = TRANSFORMS) # can also customize transform

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    for epoch in range(opt.n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):

            # SKIP BATCH SIZE OF 1
            if imgs.shape[0] <= 1: continue

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
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

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
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], os.path.join(outpath, "%d.png" % batches_done), nrow=5, normalize=True)




def run_dcgan(datapath="../data/", outpath="../tmp/"):

    INPUT_SIZE = img_shape
    SAMPLE_SIZE = 80
    NUM_LABELS = 4
    BATCH_SIZE = 8
    N_EPOCHS = 10
    NZ = opt.latent_dim
    LR = 0.01

    # create output folder
    if os.path.exists(outpath):
        shutil.rmtree(outpath)
    os.mkdir(outpath)

    # retrieve the list of image paths
    img_list = list_full_paths(datapath)


    TRANSFORMS = T.Compose([T.ToTensor(), T.Resize((opt.img_size, opt.img_size))])
    train_dataset = WoundImageDataset(img_list, transform = TRANSFORMS) # can also customize transform
    train_loader = DataLoader(train_dataset, shuffle=True,
        batch_size=BATCH_SIZE)

    model_d = dcgan.Discriminator()
    model_g = dcgan.Generator(NZ)
    criterion = nn.BCELoss()

    input = torch.FloatTensor(BATCH_SIZE, *INPUT_SIZE)
    noise = torch.FloatTensor(BATCH_SIZE, (NZ))
    
    fixed_noise = torch.FloatTensor(SAMPLE_SIZE, NZ).normal_(0,1)
    fixed_labels = torch.zeros(SAMPLE_SIZE, NUM_LABELS)

    for i in range(NUM_LABELS):
        for j in range(SAMPLE_SIZE // NUM_LABELS):
            fixed_labels[i*(SAMPLE_SIZE // NUM_LABELS) + j, i] = 1.0
    
    label = torch.FloatTensor(BATCH_SIZE)
    one_hot_labels = torch.FloatTensor(BATCH_SIZE, NUM_LABELS)

    if CUDA:
        print("hello world")
        model_d.cuda()
        model_g.cuda()
        input, label = input.cuda(), label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
        one_hot_labels = one_hot_labels.cuda()
        fixed_labels = fixed_labels.cuda()

    optim_d = torch.optim.SGD(model_d.parameters(), lr=LR)
    optim_g = torch.optim.SGD(model_g.parameters(), lr=LR)
    fixed_noise = Variable(fixed_noise)
    fixed_labels = Variable(fixed_labels)

    real_label = 1
    fake_label = 0

    for epoch_idx in range(N_EPOCHS):

        model_d.train()
        model_g.train()

        d_loss = 0.0
        g_loss = 0.0

        # train_x is the images, train_y is the labels
        for batch_idx, (train_x, train_y) in enumerate(train_loader):
            batch_size = train_x.size(0)
            #train_x = train_x.view(-1, INPUT_SIZE)

            if CUDA:
                train_x = train_x.cuda()
                train_y = train_y.cuda()

            input.resize_as_(train_x).copy_(train_x)
            label.resize_(batch_size).fill_(real_label)
            label = label.unsqueeze(1) # fix deprecation error

            #one_hot_labels.resize_(batch_size, NUM_LABELS).zero_()
            #print(one_hot_labels.shape, train_y.shape)
            #exit()
            #one_hot_labels.scatter_(1, train_y.view(batch_size, 1), 1)
            one_hot_labels = train_y
            
            inputv = Variable(input)
            labelv = Variable(label)

            output = model_d(inputv, Variable(one_hot_labels))
            optim_d.zero_grad()
            errD_real = criterion(output, labelv)
            errD_real.backward()
            realD_mean = output.data.cpu().mean()
            
            one_hot_labels.zero_()
            rand_y = torch.from_numpy(
                np.random.randint(0, NUM_LABELS, size=(batch_size,1))).cuda()
            one_hot_labels.scatter_(1, rand_y.view(batch_size,1), 1)
            noise.resize_(batch_size, NZ).normal_(0,1)
            label.resize_(batch_size).fill_(fake_label)
            label = label.unsqueeze(1) # fix deprecation error
            
            noisev = Variable(noise)
            labelv = Variable(label)
            onehotv = Variable(one_hot_labels)
            g_out = model_g(noisev, onehotv)
            output = model_d(g_out, onehotv)

            errD_fake = criterion(output, labelv)
            fakeD_mean = output.data.cpu().mean()
            errD = errD_real + errD_fake
            errD_fake.backward()
            optim_d.step()

            # train the G
            noise.normal_(0,1)
            one_hot_labels.zero_()
            rand_y = torch.from_numpy(
                np.random.randint(0, NUM_LABELS, size=(batch_size,1))).cuda()
            one_hot_labels.scatter_(1, rand_y.view(batch_size,1), 1)
            label.resize_(batch_size).fill_(real_label)
            label = label.unsqueeze(1) # fix deprecation error

            onehotv = Variable(one_hot_labels)
            noisev = Variable(noise)
            labelv = Variable(label)
            g_out = model_g(noisev, onehotv)
            output = model_d(g_out, onehotv)
            errG = criterion(output, labelv)
            optim_g.zero_grad()
            errG.backward()
            optim_g.step()
            
            d_loss += errD.item()
            g_loss += errG.item()
            if batch_idx % 10 == 0:
                print(
                "\t{} ({} / {}) mean D(fake) = {:.4f}, mean D(real) = {:.4f}".
                    format(epoch_idx, batch_idx, len(train_loader), fakeD_mean,
                        realD_mean))

                g_out = model_g(fixed_noise, fixed_labels).data.view(
                    SAMPLE_SIZE, 1, 28,28).cpu()
                save_image(g_out,
                    '{}/{}_{}.png'.format(
                        outpath, epoch_idx, batch_idx))


        print('Epoch {} - D loss = {:.4f}, G loss = {:.4f}'.format(epoch_idx,
            d_loss, g_loss))
        if epoch_idx % 2 == 0:
            torch.save({'state_dict': model_d.state_dict()},
                        '{}/model_d_epoch_{}.pth'.format(
                            outpath, epoch_idx))
            torch.save({'state_dict': model_g.state_dict()},
                        '{}/model_g_epoch_{}.pth'.format(
                            outpath, epoch_idx))


if __name__ == "__main__":
    #run_gan()
    run_dcgan()