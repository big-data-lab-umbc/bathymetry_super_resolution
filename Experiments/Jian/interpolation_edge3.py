import argparse
import os
import numpy as np
import math
import itertools
import sys
import rasterio

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

# Parameters
exp_name = "train_on_interpolation_exp7" # Experiment name used to save loss plot
hr_height=512 # Use smaller values like 128 if you have cuda memory issues 
hr_width=512  # Use smaller values like 128 if you have cuda memory issues
epoch = 1     # Set 0 if no pretrained model used
n_epochs=200
batch_size=4
lr=0.0002
b1=0.5
b2=0.999
decay_epoch=200
n_cpu=8
img_height = 128
channels=3
sample_interval=200 # How many intervals to save a training result each time
checkpoint_interval=1 # How many epoch(s) to save a trained model each time

hr_shape = (hr_height, hr_width) # The output shape

# Path to the pretrained model:
#model_pt = "/home/ubuntu/super_resolution/PyTorch-GAN/implementations/srgan/model/generator_32.pth"
model_pt = "./saved_pretrained/generator_199.pth"

# Path to training dataset:
#data_pt = "../../data2/train/"
data_pt = "../data/train/"


#--------------------#
# 1. Model Initialization
#--------------------#

# Initialize cuda
cuda = torch.cuda.is_available()
#torch.cuda.set_device(0)

# Initialize srresnet generator
generator = GeneratorResNet(in_channels=channels, out_channels=channels)
feature_extractor = FeatureExtractor()

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses definition
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()
weighted_criterion_content = torch.nn.L1Loss(reduction='none')

if cuda:
    if torch.cuda.device_count() > 1:
        generator =  torch.nn.DataParallel(generator).cuda()
        feature_extractor = torch.nn.DataParallel(feature_extractor).cuda()
        criterion_content = torch.nn.DataParallel(criterion_content).cuda()

    else:
        generator = generator.cuda()
        # discriminator = discriminator.cuda()
        feature_extractor = feature_extractor.cuda()
        # criterion_GAN = criterion_GAN.cuda()
        criterion_content = criterion_content.cuda() 

    if False:
        generator = generator.cuda()
        feature_extractor = feature_extractor.cuda()
        criterion_GAN = criterion_GAN.cuda()
        criterion_content = criterion_content.cuda()
        weighted_criterion_content = weighted_criterion_content.cuda()

#----------------------------------#
# 2. Model Loading and Data loading
#----------------------------------#

if epoch != 0:
    generator.load_state_dict(torch.load(model_pt))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    NoScaleMaskEdge(data_pt, hr_shape=hr_shape),
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_cpu,
)

#-------------------#
# 3. Model Training
#-------------------#

Gloss=[]

for epoch in range(epoch, n_epochs):
    for i, imgs in enumerate(dataloader):

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))
        input_path = imgs["input_path"]
        weight = Variable(imgs["weight"].type(Tensor))
        mask_ocean = Variable(imgs["mask_ocean"].type(Tensor))
        mask_edge = Variable(imgs["mask_edge"].type(Tensor))

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr) 


        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)

        edge_gen = mask_edge * gen_hr
        edge_hr = imgs_hr * mask_edge

        edge_gen_features = feature_extractor(edge_gen)
        edge_features = feature_extractor(edge_hr)

        def weighted_content_loss(predict, label, weight):
            loss_vector = weighted_criterion_content(predict, label)
            loss_weight = loss_vector * weight
            return loss_weight.sum() / weight.sum()
        
        loss_content = criterion_content(gen_hr, imgs_hr.detach()) + criterion_content(gen_features, real_features.detach())
        loss_edge = criterion_content(edge_gen, edge_hr.detach()) + criterion_content(edge_gen_features, edge_features.detach())
        
        if mask_edge.sum() != 0:
            loss_G = loss_content.mean() + loss_edge.mean()
        else:
            loss_G = loss_content.mean()


        loss_G.backward()
        optimizer_G.step()


        # --------------
        #  Log Progress
        # --------------

        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [G loss: %f] [Edge loss: %f]\n"
            % (epoch+1, n_epochs, i+1, len(dataloader), loss_G.item(), loss_edge.mean().item())
        )

        # Dloss.append(loss_D.item())
        Gloss.append(loss_G.item())

        batches_done = epoch * len(dataloader) + i
        if False:
            # --------------
            #  Save results
            # --------------

            if (batches_done+1) % sample_interval == 0:
                # Save image grid with upsampled inputs and SRGAN outputs
                # Inverse normalization (min, max, mean and var are calculated by another norlization script)
                max = 6787
                min = -10802
                mean = 0.5
                var = 0.5
                gen_hr_single = gen_hr.cpu().detach().numpy() * var + mean
                gen_hr2_single = gen_hr_single*(max - min) + min


                # --------------
                #  Save TIFF 
                #  There're 4 images in one batch, here the 1st and 3rd results in the batch are saved
                # --------------
                hr_root, hr_name = os.path.split(input_path[0])
                with rasterio.Env():
                    with rasterio.open(input_path[0]) as src: 
                        profile = src.meta
                        print(profile)
                        t = src.transform * src.transform.scale(1/4, 1/4)
                    
                        profile.update(
                            width = hr_width, 
                            height = hr_height,
                            dtype = rasterio.float32,
                            transform = t 
                        )
                        #print(profile)
                    with rasterio.open("images/noscale14/{}_{}.tif".format(hr_name,batches_done), 'w', **profile) as dst:
                        dst.write(gen_hr2_single[0][0], indexes = 1)

                hr_root, hr_name = os.path.split(input_path[2])
                with rasterio.Env():
                    with rasterio.open(input_path[2]) as src: 
                        profile = src.meta
                        print(profile)
                        t = src.transform * src.transform.scale(1/4, 1/4)
                    
                        profile.update(
                            width = hr_width, 
                            height = hr_height,
                            dtype = rasterio.float32,
                            transform = t 
                        )
                        #print(profile)
                    with rasterio.open("images/noscale14/{}_{}.tif".format(hr_name,batches_done), 'w', **profile) as dst:
                        dst.write(gen_hr2_single[2][0], indexes = 1)


            # ---------------------
            #  Save img grid as PNG 
            #  Img grid: lr, gen_hr and hr
            #  (from left column to right)
            # ----------------------

            # imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            imgs_hr = make_grid(imgs_hr, nrow=1,normalize=True)
            img_grid = torch.cat((imgs_lr, gen_hr, imgs_hr), -1)

            save_image(img_grid, "images/noscale14/%d.png" % batches_done, normalize=False) 


    # ------------------
    #  Save models
    # ------------------
    if checkpoint_interval != -1 and (epoch+1) % checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models_exp6/generator_%d.pth" % epoch)
        # torch.save(discriminator.state_dict(), "saved_models/baseline1_srgan/discriminator_%d.pth" % epoch)


#-------------------#
# 4. Loss plot saving
#-------------------#
from matplotlib import pyplot

#np.save('Loss/{}.npy'.format(exp_name),Gloss)

#pyplot.figure(figsize=(16,9))
#pyplot.plot(Gloss, label='Generator')
#pyplot.legend()
#pyplot.savefig('Loss/{}.png'.format(exp_name)) 
#pyplot.show()