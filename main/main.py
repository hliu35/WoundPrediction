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
from synth_labels import synthesize_softmax_labels as synth_softmax
from synth_labels import synthesize_onehot_labels as synth_onehot



parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches") # changed from 16 to 8
parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate") # changed from 0.0002
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space") # original: 100, new: 16
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")  # changed from 64 to 128
parser.add_argument('--n_classes', type=int, default=4, help='number of classes for dataset')
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=50, help="interval betwen image samples")
opt = parser.parse_args()


img_shape = (opt.channels, opt.img_size, opt.img_size)
CUDA = True if torch.cuda.is_available() else False



def list_full_paths(directory):
    full_list = [os.path.join(directory, file) for file in os.listdir(directory) if "png" in file]
    return sorted(full_list)


def run_gan(datapath, annotation_file, outpath="../tmp/"):

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

    for epoch in range(opt.n_epochs):
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




def run_cgan(datapath, annotation_file, outpath="../tmp/"):

    img_shape = (opt.channels, opt.img_size, opt.img_size)
    n_classes = opt.n_classes

    # create output folder
    if os.path.exists(outpath):
        shutil.rmtree(outpath)
    os.mkdir(outpath)

    # retrieve the list of image paths
    img_list = list_full_paths(datapath)
    #print(img_list[:10])

        
    # Loss function
    adversarial_loss = torch.nn.BCELoss()  ######## TEST with MSE
    #adversarial_loss = torch.nn.MSELoss()


    # Initialize Generator and discriminator
    generator = dcgan.Generator(img_shape, opt.latent_dim, opt.n_classes)
    discriminator = dcgan.Discriminator(img_shape, opt.n_classes)

    # Initialize weights
    #generator.apply(weights_init_normal)
    #discriminator.apply(weights_init_normal)

    if CUDA:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Configure data loader and compose transform functions
    TRANSFORMS = T.Compose([T.ToTensor(), T.Resize((opt.img_size, opt.img_size))])
    dataset = WoundImageDataset(img_list, \
        annotation_file,\
        transform = TRANSFORMS) # can also customize transform

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    #optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)            ########## optimizer changed from default
    #optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)


    Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor


    batches_done=0
    for epoch in range(opt.n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):

            batch_size = opt.batch_size

            # SKIP BATCH SIZE OF 1
            if imgs.shape[0] < batch_size: continue

            # Adversarial ground truths
            valid = Variable(torch.ones(batch_size).cuda(), requires_grad=False)
            fake = Variable(torch.zeros(batch_size).cuda(), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(torch.FloatTensor).cuda())

            #real_y = torch.zeros(batch_size, n_classes)
            #real_y = Variable(real_y.scatter_(1, labels.view(batch_size, n_classes), 1).cuda())
            real_y = labels.cuda()

            #y = Variable(y.cuda())

            # Sample noise and labels as generator input
            noise = Variable(torch.randn((batch_size, opt.latent_dim)).cuda())
            #gen_labels = (torch.rand(batch_size, 1) * n_classes).type(torch.LongTensor)
            #gen_y = torch.zeros(batch_size, n_classes)
            #gen_y = Variable(gen_y.scatter_(1, gen_labels.view(batch_size, 1), 1).cuda())
            
            gen_y = synth_softmax(n_classes=opt.n_classes, batch_size=batch_size).to("cuda")
            #gen_y = synth_onehot(n_classes=opt.n_classes, batch_size=batch_size, fixed=False).to("cuda")
            #print(gen_y)
            #exit()
            
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            #gen_imgs = generator(noise, gen_y)
            # Loss measures generator's ability to fool the discriminator
            gen_imgs = generator(noise, gen_y)
            g_loss = adversarial_loss(discriminator(gen_imgs,gen_y).squeeze(), valid)

            g_loss.backward()
            optimizer_G.step()


            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Loss for real images
            d_real_loss = adversarial_loss(discriminator(real_imgs, real_y).squeeze(), valid)
            # Loss for fake images
            
            d_fake_loss = adversarial_loss(discriminator(gen_imgs.detach(),gen_y).squeeze(), fake)
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss)

            d_loss.backward()
            optimizer_D.step()


            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                                d_loss.data.cpu(), g_loss.data.cpu()))

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                #noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))).cuda())
                noise = Variable(torch.randn((batch_size, opt.latent_dim)).cuda())

                # fixed labels
                #y_ = torch.LongTensor(np.array([num for num in range(n_classes)])).view(n_classes,1).expand(-1,n_classes).contiguous()
                #y_fixed = torch.zeros(n_classes**2, n_classes)
                #y_fixed = Variable(y_fixed.scatter_(1,y_.view(n_classes**2,1),1).cuda())
                y_fixed = synth_onehot(n_classes=opt.n_classes, batch_size=batch_size, fixed=True).to("cuda")

                gen_imgs = generator(noise, y_fixed).view(-1, *img_shape)
                save_image(gen_imgs.data, os.path.join(outpath, '%d-%d.png' % (epoch,batches_done)), nrow=n_classes, normalize=True) # nrow = number of img per row

        torch.save(generator, os.path.join(outpath, "cgan_gen.pth"))






if __name__ == "__main__":
    print(opt)
    assert(opt.batch_size > 1)
    
    # configure data and annotation path
    datapath = "../data_augmented/"
    annotation_file = "../data_augmented/augmented_labels.csv"

    # train/test the models
    run_gan(datapath, annotation_file)
    #run_cgan(datapath, annotation_file)

    # custom test scripts
    
    # generator = torch.load("../tmp/cgan_gen.pth")

    # noise = Variable(torch.randn((opt.batch_size, opt.latent_dim)).cuda())
    # stage = 2
    # y_fixed = torch.ones((opt.batch_size, opt.n_classes)).to("cuda")
    # y_fixed[:, stage] = 1

    # gen_imgs = generator(noise, y_fixed).view(-1, *img_shape)
    # save_image(gen_imgs.data, os.path.join("../tmp/", 'test_stage_%d.png' % (stage)), nrow=opt.batch_size, normalize=True) # nrow = number of img per row