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



epoch=1
n_epochs=200
#dataset_name="img_hr"
batch_size=4
lr=0.0002
b1=0.5
b2=0.999
decay_epoch=200
n_cpu=8
img_height = 128
hr_height=512
hr_width=512
channels=3
sample_interval=400
checkpoint_interval=1

cuda = torch.cuda.is_available()

hr_shape = (hr_height, hr_width)

# Initialize generator and discriminator
generator = GeneratorResNet(in_channels=channels, out_channels=channels)
# discriminator = Discriminator(input_shape=(channels, *hr_shape))
feature_extractor = FeatureExtractor()

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

if cuda:
    generator = generator.cuda()
    # discriminator = discriminator.cuda()
    feature_extractor = feature_extractor.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()

if epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("/home/ubuntu/super_resolution/PyTorch-GAN/implementations/srgan/model/generator_200.pth"))
    # discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth"))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    ImageDatasetPretrainTiff("../../data2/train/", hr_shape=hr_shape),
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_cpu,
)

# ----------
#  Training
# ----------

Dloss=[]
Gloss=[]

for epoch in range(epoch, n_epochs):
    for i, imgs in enumerate(dataloader):

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))
        input_path = imgs["input_path"]

        # Adversarial ground truths
        # valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        # fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr) 

        # Adversarial loss
        # loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_hr, imgs_hr.detach()) + criterion_content(gen_features, real_features.detach())

        # Total loss
        loss_G = loss_content

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # optimizer_D.zero_grad()

        # Loss of real and fake images
        # loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        # loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

        # Total loss
        # loss_D = (loss_real + loss_fake) / 2

        # loss_D.backward()
        # optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------


        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [G loss: %f]\n"
            % (epoch, n_epochs, i+1, len(dataloader), loss_G.item())
        )

        Gloss.append(loss_G.item())

        batches_done = epoch * len(dataloader) + i

        if (batches_done+2) % sample_interval == 0:
            # Save image grid with upsampled inputs and SRGAN outputs
            # Inverse normalization normalization
            gen_hr_single = gen_hr.cpu().detach().numpy()*0.5+0.5
            gen_hr2_single = gen_hr_single*(6787+10802)-10802

            hr_root, hr_name = os.path.split(input_path[0])
            with rasterio.Env():
                with rasterio.open(input_path[0]) as src: 
                    profile = src.meta
                    print(profile)
                    t = src.transform * src.transform.scale(1/4, 1/4)
                
                    profile.update(
                        width = 128 * 4, 
                        height = img_height * 4,
                        dtype = rasterio.float32,
                        transform = t 
                    )
                    # print(profile)
                with rasterio.open("images/baseline2/gen_{}_{}.tif".format(batches_done,hr_name[:-10]), 'w', **profile) as dst:
                    dst.write(gen_hr2_single[0][0], indexes = 1)
                    
            hr_root, hr_name = os.path.split(input_path[2])  
            with rasterio.Env():
                with rasterio.open(input_path[2]) as src: 
                    profile = src.meta
                    print(profile)
                    t = src.transform * src.transform.scale(1/4, 1/4)
                
                    profile.update(
                        width = 128 * 4, 
                        height = img_height * 4,
                        dtype = rasterio.float32,
                        transform = t 
                    )
                with rasterio.open("images/baseline2/gen_{}_{}.tif".format(batches_done,hr_name[:-10]), 'w', **profile) as dst:
                    dst.write(gen_hr2_single[2][0], indexes = 1)
            

            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            imgs_hr = make_grid(imgs_hr, nrow=1,normalize=True)
            img_grid = torch.cat((imgs_lr, gen_hr, imgs_hr), -1)
            
            # gen_hr = make_grid(gen_hr, nrow=4, normalize=True)
            # imgs_hr = make_grid(imgs_hr, nrow=4,normalize=True)
            # img_grid = torch.cat((gen_hr, imgs_hr), -1)
            
            save_image(img_grid, "images/baseline2/%d.png" % batches_done, normalize=False) #save whole table: lr, gen_hr and hr

            # print("batch_done:%d\n" %batches_done)
  
    if checkpoint_interval != -1 and (epoch+1) % checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/baseline2/generator_%d.pth" % epoch)
        # torch.save(discriminator.state_dict(), "saved_models/1_Jan25/discriminator_%d.pth" % epoch)

from matplotlib import pyplot
np.save('Loss/baseline2.npy',Gloss)

pyplot.figure(figsize=(16,9))
pyplot.ylim(0,0.2)
pyplot.plot(Gloss, label='Generator')
pyplot.legend()
pyplot.savefig('baseline2.png') 
pyplot.show()