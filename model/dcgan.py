import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

# https://github.com/TeeyoHuang/conditional-GAN

# https://arxiv.org/pdf/2104.00567.pdf

CUDA = True if torch.cuda.is_available() else False

# some parameters
N_CHANNELS = 3 # input channels (r, g, b) = 3


class Generator(nn.Module):
    # initializers
    def __init__(self, img_shape, latent_dim, n_classes):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        self.fc1_1 = nn.Linear(self.latent_dim, 256)
        self.fc1_1_bn = nn.BatchNorm1d(256, 0.8) # test w/o batchnorm
        self.fc1_2 = nn.Linear(self.n_classes, 256) 
        self.fc1_2_bn = nn.BatchNorm1d(256, 0.8) # test w/o batchnorm
        
        #self.fc2 = nn.Linear(512, 1024)
        #self.fc2_bn = nn.BatchNorm1d(1024, 0.8)
        #self.fc3 = nn.Linear(1024, 2048)
        #self.fc3_bn = nn.BatchNorm1d(2048, 0.8)
        #self.fc4 = nn.Linear(2048, int(np.prod(self.img_shape)))

        # changes to 2d
        nz = int(64 * img_shape[1]//2//2 *img_shape[2]//2//2)
        self.fc = nn.Linear(512, nz)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)




    # forward method
    def forward(self, z, label):
        #x = F.relu(self.fc1_1_bn(self.fc1_1(z))) # w/ batchnorm
        #y = F.relu(self.fc1_2_bn(self.fc1_2(label))) # w/ batchnorm
        x = F.leaky_relu(self.fc1_1(z), 0.2) # test: w/o batchnorm, leaky
        y = F.leaky_relu(self.fc1_2(label), 0.2) # test: w/o batchnorm, leaky
        x = torch.cat([x, y], 1)

        #x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2)
        #x = F.leaky_relu(self.fc3_bn(self.fc3(x)), 0.2)
        #x = F.tanh(self.fc4(x))
        
        # 2d
        x = self.fc(x)
        x = x.view(-1, 64, self.img_shape[1]//2//2, self.img_shape[2]//2//2)
        x = F.leaky_relu(self.bn1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.deconv1(x)), 0.2)
        x = F.tanh(self.deconv2(x))

         
        return x


class Discriminator(nn.Module):
    # initializers
    def __init__(self, img_shape, n_classes):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.n_classes = n_classes

        #self.fc1_1 = nn.Linear(int(np.prod(self.img_shape)), 512) # original 1024, new 512, working: 256
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding=1, padding_mode="zeros") # outsize: insize + 2*padding - kernel + 1
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding=1, padding_mode="zeros") # outsize: insize + 2*padding - kernel + 1
        #18 * 64 * 64 input features, 512 output features (see sizing flow below)

        self.fc1_1 = nn.Linear(int(32 * img_shape[1]//2//2 * img_shape[2]//2//2), 512)
        self.fc1_2 = nn.Linear(self.n_classes, 512)
        self.fc2 = nn.Linear(1024, 512) # original 1024+1024, new 512+512
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 1)


    # forward method
    def forward(self, input, label):
        # additional convs    
        #x = F.relu(self.conv1(input))
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = self.pool(x)
        #x = F.relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = self.pool(x)


        #x = F.leaky_relu(self.fc1_1(input.view(input.size(0),-1)), 0.2)
        x = F.leaky_relu(self.fc1_1(x.view(x.size(0),-1)), 0.2)
        y = F.leaky_relu(self.fc1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        #x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2) # with batchnorm
        #x = F.leaky_relu(self.fc3_bn(self.fc3(x)), 0.2) # with batchnorm
        x = F.leaky_relu(self.fc2(x), 0.2) # test w/o batchnorm
        x = F.leaky_relu(self.fc3(x), 0.2) # test w/o batchnorm
        validity = torch.sigmoid(self.fc4(x))
        return validity




if __name__ == '__main__':

    img_save_path = 'images'
    os.makedirs(img_save_path, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--beta2', type=float, default=0.999, help='adam: decay of second order momentum of gradient')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
    parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--sample_interval', type=int, default=200, help='interval between image sampling')
    opt = parser.parse_args()
    print(opt)

    img_shape = (opt.channels, opt.img_size, opt.img_size)
    latent_dim = opt.latent_dim

        
    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize Generator and discriminator
    generator = Generator(img_shape, latent_dim)
    discriminator = Discriminator(img_shape=img_shape)

    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Initialize weights
    #generator.apply(weights_init_normal)
    #discriminator.apply(weights_init_normal)

    # Configure data loader
    os.makedirs('../../data', exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('../../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.Resize((opt.img_size, opt.img_size)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5])
                    ])),
        batch_size=opt.batch_size, shuffle=True, drop_last=True)
    print('the data is ok')

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))


    batches_done=0
    for epoch in range(opt.n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):

            Batch_Size = opt.batch_size
            N_Class = opt.n_classes
            # Adversarial ground truths
            valid = Variable(torch.ones(Batch_Size).cuda(), requires_grad=False)
            fake = Variable(torch.zeros(Batch_Size).cuda(), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(torch.FloatTensor).cuda())

            real_y = torch.zeros(Batch_Size, N_Class)
            real_y = Variable(real_y.scatter_(1, labels.view(Batch_Size, 1), 1).cuda())
            #y = Variable(y.cuda())

            # Sample noise and labels as generator input
            noise = Variable(torch.randn((Batch_Size, opt.latent_dim)).cuda())
            gen_labels = (torch.rand(Batch_Size, 1) * N_Class).type(torch.LongTensor)
            gen_y = torch.zeros(Batch_Size, N_Class)
            gen_y = Variable(gen_y.scatter_(1, gen_labels.view(Batch_Size, 1), 1).cuda())

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Loss for real images
            d_real_loss = adversarial_loss(discriminator(real_imgs, real_y).squeeze(), valid)
            # Loss for fake images
            gen_imgs = generator(noise, gen_y)
            d_fake_loss = adversarial_loss(discriminator(gen_imgs.detach(),gen_y).squeeze(), fake)
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss)

            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            #gen_imgs = generator(noise, gen_y)
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs,gen_y).squeeze(), valid)

            g_loss.backward()
            optimizer_G.step()


            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                                d_loss.data.cpu(), g_loss.data.cpu()))

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (N_Class**2, opt.latent_dim))).cuda())
                #fixed labels
                y_ = torch.LongTensor(np.array([num for num in range(N_Class)])).view(N_Class,1).expand(-1,N_Class).contiguous()
                y_fixed = torch.zeros(N_Class**2, N_Class)
                y_fixed = Variable(y_fixed.scatter_(1,y_.view(N_Class**2,1),1).cuda())

                gen_imgs = generator(noise, y_fixed).view(-1, *img_shape)

                save_image(gen_imgs.data, img_save_path + '/%d-%d.png' % (epoch,batches_done), nrow=N_Class, normalize=True)





