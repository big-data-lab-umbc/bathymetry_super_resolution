# Use a pretrianed model to continue training, with only water-content loss
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

torch.cuda.set_device(1)

cuda = torch.cuda.is_available()

hr_shape = (hr_height, hr_width)

# Initialize generator
generator = GeneratorResNet(in_channels=channels, out_channels=channels)
feature_extractor = FeatureExtractor()

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()
weighted_criterion_content = torch.nn.L1Loss(reduction='none')

if cuda:
    generator = generator.cuda()
    feature_extractor = feature_extractor.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()
    weighted_criterion_content = weighted_criterion_content.cuda()

if epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("/home/ubuntu/PyTorch-GAN/implementations/srgan/model/generator_200.pth"))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    ImageDatasetBaseline2_Water("../../data2/train/", hr_shape=hr_shape),
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
        # mask_hr = Variable(imgs["mask_hr"].type(Tensor))
        input_path = imgs["input_path"]
        weight = Variable(imgs["weight"].type(Tensor))
        mask = Variable(imgs["mask"].type(Tensor))

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr) 
        
        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        # mask_real_features = feature_extractor(mask_hr)

        def weighted_content_loss(predict, label, weight):
            loss_vector = weighted_criterion_content(predict, label)
            loss_weight = loss_vector * weight
            return loss_weight.sum() / weight.sum()
        
        # 2 parts of masked loss: WITHOUT weight and WITH weight
        # water_content = weighted_content_loss(gen_hr, imgs_hr.detach(), mask.detach())
        # weighted_content = weighted_content_loss(gen_hr, imgs_hr.detach(), weight.detach())

        # 1. Baseline loss: water mask has been applied
        loss_content = criterion_content(gen_hr, imgs_hr.detach()) + criterion_content(gen_features, real_features.detach())
        
        # 2. Maksed loss WITHOUT weight:
        # loss_content = water_content
        
        # 3. Masked loo WITH weight:
        # loss_content = weighted_content
        
        # 4. Maksed loss of both:
        # loss_content = 0.5 * water_content + 0.5 * weighted_content
        
        # Total loss
        loss_G = loss_content

        loss_G.backward()
        optimizer_G.step()

        # --------------
        #  Log Progress
        # --------------


        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [G loss: %f]\n"
            % (epoch, n_epochs, i+1, len(dataloader), loss_G.item())
        )

        Gloss.append(loss_G.item())

        batches_done = epoch * len(dataloader) + i

        if (batches_done+1) % sample_interval == 0:
            # Save image grid with upsampled inputs and SRGAN outputs
            # Inverse normalization normalization
            gen_hr_single = gen_hr.cpu().detach().numpy()*0.5+0.5
            gen_hr2_single = gen_hr_single*(6787+10802)-10802

            mask = mask.cpu().detach().numpy()
        
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
                with rasterio.open("images/baseline2_water/gen_{}_{}.tif".format(batches_done,hr_name[:-10]), 'w', **profile) as dst:
                    dst.write(gen_hr2_single[0][0]*mask[0][0], indexes = 1)
            
            hr_root, hr_name = os.path.split(input_path[1])  
            with rasterio.Env():
                with rasterio.open(input_path[1]) as src: 
                    profile = src.meta
                    print(profile)
                    t = src.transform * src.transform.scale(1/4, 1/4)
                
                    profile.update(
                        width = 128 * 4, 
                        height = img_height * 4,
                        dtype = rasterio.float32,
                        transform = t 
                    )
                with rasterio.open("images/baseline2_water/gen_{}_{}.tif".format(batches_done,hr_name[:-10]), 'w', **profile) as dst:
                    dst.write(gen_hr2_single[1][0]*mask[1][0], indexes = 1)
                
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
                with rasterio.open("images/baseline2_water/gen_{}_{}.tif".format(batches_done,hr_name[:-10]), 'w', **profile) as dst:
                    dst.write(gen_hr2_single[2][0]*mask[2][0], indexes = 1)
            
            hr_root, hr_name = os.path.split(input_path[3])  
            with rasterio.Env():
                with rasterio.open(input_path[3]) as src: 
                    profile = src.meta
                    print(profile)
                    t = src.transform * src.transform.scale(1/4, 1/4)
                
                    profile.update(
                        width = 128 * 4, 
                        height = img_height * 4,
                        dtype = rasterio.float32,
                        transform = t 
                    )
                with rasterio.open("images/baseline2_water/gen_{}_{}.tif".format(batches_done,hr_name[:-10]), 'w', **profile) as dst:
                    dst.write(gen_hr2_single[3][0]*mask[3][0], indexes = 1)

            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            imgs_hr = make_grid(imgs_hr, nrow=1,normalize=True)
            img_grid = torch.cat((imgs_lr, gen_hr, imgs_hr), -1)
            
            save_image(img_grid, "images/baseline2_water/%d.png" % batches_done, normalize=False) #save whole table: lr, gen_hr and hr

            # print("batch_done:%d\n" %batches_done)
  
    if checkpoint_interval != -1 and (epoch+1) % checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/baseline2_water/generator_%d.pth" % epoch)


from matplotlib import pyplot
np.save('Loss/baseline2_water.npy',Gloss)

pyplot.figure(figsize=(16,9))
pyplot.plot(Gloss, label='Generator')
pyplot.legend()
pyplot.savefig('baseline2_water.png') 
pyplot.show()