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
parser.add_argument("--epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches") # changed from 16 to 8
parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate") # changed from 0.0002
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space") # original: 100, new: 16
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")  # changed from 64 to 128
parser.add_argument('--n_classes', type=int, default=16, help='number of classes for dataset')
#parser.add_argument('--n_classes', type=int, default=16, help='number of classes for dataset')
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=200, help="interval betwen image samples")
opt = parser.parse_args()


IMG_SHAPE = (opt.channels, opt.img_size, opt.img_size)
CUDA = True if torch.cuda.is_available() else False
CLIP = 0.5
D_UPDATE_THRESHOLD = 0.8



def list_full_paths(directory):
    full_list = [os.path.join(directory, file) for file in os.listdir(directory) if "png" in file]
    random.shuffle(full_list)
    #test_list = [x for x in full_list if "Day 15" in x] # test with only day 15 data
    return full_list
    #return test_list

def list_full_paths_combs(directory, mode="train"):

    full_list = [os.path.join(directory, file) for file in os.listdir(directory) if "png" in file]
    test_imgs = [x for x in full_list if "Y8-4-L" in x or "A8-1-R" in x]
    val_imgs = [x for x in full_list if "Y8-4-R" in x or "A8-1-L" in x]
    train_imgs = set(full_list).difference(set(test_imgs))
    train_imgs = train_imgs.difference(set(val_imgs))
    train_imgs = list(train_imgs)

    if mode=="train": full_list = train_imgs
    elif mode=="val": full_list = val_imgs
    else: full_list = test_imgs

    Ids = []
    for i in range(16):
        temp = full_list[i].split("/")[-1]
        Ids.append(temp.split("_")[1])

    seperated_Ids = []
    for id in Ids:
        id_list = []
        for png in full_list:
            if id in png:
                id_list.append(png)
        for i in range(6):
            id_list.append(id_list.pop(1))
        seperated_Ids.append(id_list)

    combs = []
    for i in range(16):
        combs.append([comb for comb in combinations(seperated_Ids[i], 3)])

    combs_list = [item for sublist in combs for item in sublist]
    random.shuffle(combs_list)
    return combs_list


def train_gan(datapath, annotation_file, outpath="../tmp/"):
    ''' This is the script to train Vanilla GAN '''
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




def train_cgan(datapath, annotation_file, outpath="../tmp/"):
    ''' This is the script to train Conditional GAN a.k.a. DCGAN '''
    # https://www.reddit.com/r/MachineLearning/comments/5asl74/discussion_discriminator_converging_to_0_loss_in/

    C = opt.n_classes
    if C not in [4, 16]: raise NotImplementedError("Check n_classes in arguments")

    # create output folder
    if os.path.exists(outpath):
        shutil.rmtree(outpath)
    os.mkdir(outpath)

    # retrieve the list of image paths
    #img_list = list_full_paths(datapath)
    train_imgs = list_full_paths_combs(datapath, "train")
    #val_imgs = list_full_paths_combs(datapath, "val")
    #test_imgs = list_full_paths_combs(datapath, "test")


    # Loss functions for part 1, 2, and 3
    adversarial_loss = torch.nn.BCELoss()
    #adversarial_loss = torch.nn.MSELoss() # an alternative loss function
    
    if C == 16:
        # if it's 16, they are not binary values and should be treated as random numbers
        embedding_loss = torch.nn.MSELoss() # embedding loss with MSE
        #embedding_loss = torch.nn.CosineEmbeddingLoss() # embedding loss with CEL https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html
    else:
        # if it's 4 (stages), then it must be binary cross entropy
        embedding_loss = torch.nn.BCELoss()
    
    temporal_loss = torch.nn.BCELoss()


    # Initialize Generator and discriminator
    generator = dcgan.Generator(IMG_SHAPE, opt.latent_dim, C)
    discriminator = dcgan.Discriminator(IMG_SHAPE, C)


    # Initialize Temporal Encoder
    temporal_encoder = TE.loadClassifier("../model/best_classifier.tar")
    for p in temporal_encoder.parameters():
        p.require_grads = False
    temporal_encoder.eval()

    # Initialize Temporal Discriminator
    tc_latent_dim = 32
    temporal_classifier = TC.TemporalClassifier(IMG_SHAPE, tc_latent_dim)


    # Initialize weights
    #generator.apply(weights_init_normal)
    #discriminator.apply(weights_init_normal)

    
    if CUDA:
        generator.cuda()
        discriminator.cuda()
        temporal_encoder.cuda()
        temporal_classifier.cuda()

        adversarial_loss.cuda()
        embedding_loss.cuda()
        temporal_loss.cuda()


    # Configure data loaders and compose transform functions
    TRANSFORMS = T.Compose([T.ToTensor(), T.Resize((opt.img_size, opt.img_size))])


    train_dataset = WoundImagePairsDataset(train_imgs, annotation_file, transform = TRANSFORMS)
    #val_dataset = WoundImagePairsDataset(val_imgs, annotation_file, transform = TRANSFORMS)
    #test_dataset = WoundImagePairsDataset(test_imgs, annotation_file, transform = TRANSFORMS)

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    #val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)
    #test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)


    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


    # densenet image preparation
    DENSENET_IMAGE_SHAPE = 244
    transform_densenet = T.Compose([T.Resize(DENSENET_IMAGE_SHAPE), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    batches_done=0
    for epoch in range(opt.epochs):

        for i, (imgs_i, imgs_j, imgs_k, Y4, Y16) in enumerate(train_dataloader):
            # training parameters
            B = opt.batch_size

            # SKIP BATCH SIZE OF 1
            if imgs_k.shape[0] < B: continue


            # Sample labels as generator input
            if C == 4:
                real_y = Y4.cuda()
                gen_y = synth_softmax(n_classes=C, batch_size=B).to("cuda")
                #gen_y = synth_onehot(n_classes=C, batch_size=B, sorted=False).to("cuda")
                y_disp = synth_onehot(n_classes=C, batch_size=B, sorted=True).to("cuda") # labels for display
            
            elif C == 16:
                real_y = Y16.cuda()
                gen_y = Y16.cuda()
                y_disp = Y16.cuda() # labels for display
            

            # sample noise for generating fakes
            noise = Variable(torch.randn((B, opt.latent_dim)).cuda())

            # Adversarial ground truths
            valid = Variable(torch.ones(B).cuda(), requires_grad=False)
            fake = Variable(torch.zeros(B).cuda(), requires_grad=False)

            # Configure input
            imgs_k = Variable(imgs_k.type(torch.FloatTensor).cuda())
            imgs_i = Variable(imgs_i.type(torch.FloatTensor).cuda())
            imgs_j = Variable(imgs_j.type(torch.FloatTensor).cuda())
            #real_y = torch.zeros(batch_size, n_classes)
            #real_y = Variable(real_y.scatter_(1, labels.view(batch_size, n_classes), 1).cuda())


            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            #gen_imgs = generator(noise, gen_y)

            # Loss measures generator's ability to fool the discriminator
            gen_imgs = generator(noise, gen_y).view(B, *IMG_SHAPE)
            prediction = discriminator(gen_imgs, gen_y).squeeze()
            g_loss = adversarial_loss(prediction, valid)

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), CLIP) # notice the trailing _ representing in-place
            optimizer_G.step()


            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # REALISM LOSS (Loss R)

            # Loss for real images
            d_real_loss = adversarial_loss(discriminator(imgs_k, real_y).squeeze(), valid)
            # Loss for fake images
            d_fake_loss = adversarial_loss(discriminator(gen_imgs.detach(),gen_y).squeeze(), fake)
            
            
            
            # EMBEDDING LOSS (Loss yhat|z)

            cls_input = gen_imgs.detach()
            cls_input = transform_densenet(cls_input)

            target = gen_y

            label_pred, fake_embeddings = temporal_encoder(cls_input)         # NOT SURE IF DETACHING AGAIN IS CORRECT FOR LOSS COMPUTATION
            prediction = F.softmax(label_pred) if C == 4 else fake_embeddings # label_pred has not been softmaxed yet

            emb_loss = embedding_loss(target, prediction)

            
            # TEMPORAL COHERENCE LOSS (Loss Xpred_k | Xi, Xj)
            Yirk = torch.ones((B,1)) # i and real k
            Yjrk = torch.ones((B,1)) # j and real k
            Yifk = torch.zeros((B,1)) # i and fake k
            Yjfk = torch.zeros((B,1)) # j and fake k

            gen_k = gen_imgs.detach()

            temporal_target = torch.cat((Yirk, Yjrk, Yifk, Yjfk), dim=0).cuda()
            temporal_K = torch.cat((imgs_k, imgs_k, gen_k, gen_k), dim=0)
            temporal_IJ = torch.cat((imgs_i, imgs_j, imgs_i, imgs_j), dim=0)

            temporal_pred = temporal_classifier(temporal_IJ, temporal_K)

            t_loss = temporal_loss(temporal_pred, temporal_target)

            
            # TOTAL DISCRIMINATOR LOSS (L)

            #d_loss = (d_real_loss + d_fake_loss)
            d_loss = (d_real_loss + d_fake_loss + emb_loss + t_loss)

            
            # conditional update D
            #if d_loss >= D_UPDATE_THRESHOLD:
            if g_loss <= 10.0:   # TEST
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), CLIP) # notice the trailing _ representing in-place
                optimizer_D.step()

            
            # print results
            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.epochs, i, len(train_dataloader),
                                                                d_loss.data.cpu(), g_loss.data.cpu()))

            batches_done = epoch * len(train_dataloader) + i
            if batches_done % opt.sample_interval == 0:
                #noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))).cuda())
                noise = Variable(torch.randn((B, opt.latent_dim)).cuda())

                # y_disp was defined at the very beginning of training
                #print(y_disp)
                gen_imgs = generator(noise, y_disp).view(-1, *IMG_SHAPE)
                save_image(gen_imgs.data, os.path.join(outpath, '%d-%d.png' % (epoch,batches_done)), nrow=C//4, normalize=True) # nrow = number of img per row, original C, current C//4
                #save_image(imgs_k.data, os.path.join(outpath, '%d-%d.png' % (epoch,batches_done)), nrow=C//2, normalize=True) # real data

        if epoch % 10 == 0:
            torch.save(generator, os.path.join(outpath, "cgan_gen.pth"))



    #return {"model":[generator, discriminator], "dataloaders":[train_dataloader, val_dataloader, test_dataloader]}




if __name__ == "__main__":
    print(opt)
    assert(opt.batch_size > 1)
    
    # configure data and annotation path
    datapath = "../data_augmented/"
    #annotation_file = "../data_augmented/augmented_labels.csv"
    annotation_file = "../data_augmented/augmented_data.csv"

    # train/test the models
    #train_gan(datapath, annotation_file)
    #train_cgan(datapath, annotation_file)
    l = list_full_paths_combs(datapath, "train")
    print(l)