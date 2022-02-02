"""
Single-Image Super-resolution Generator (for testing)

Instrustion on running the script:
1. Download test data to ../../data/test
2. change 'img_height' and 'img_width' with the size of test data
4. Run the sript using command 'python3 srgan.py'
"""

import argparse
import os
from PIL.Image import Image
import numpy as np
import math
import itertools
import sys
import rasterio
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch



parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of input image height")
parser.add_argument("--img_width", type=int, default=128, help="size of input image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--source_path", type=str, default="/home/ubuntu/PyTorch-GAN/data/test", help="path of the source data")
parser.add_argument("--saved_path", type=str, default="/home/ubuntu/PyTorch-GAN/data/test/results", help="path of the generated data")
parser.add_argument("--model", type=str, default="/home/ubuntu/PyTorch-GAN/implementations/srgan/model/generator_59.pth", help="path of the generator")

opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available()

input_shape = (opt.img_height, opt.img_width)

# Initialize generator
generator = GeneratorResNet(in_channels=opt.channels, out_channels=opt.channels)




if cuda:
    generator = generator.cuda()

generator.load_state_dict(torch.load(opt.model))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

img_shape = (opt.img_height, opt.img_width)


dataloader = DataLoader(
    ImageDatasetPretrainTest(opt.source_path, img_shape = img_shape),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,
)

print('Generating')
print(len(dataloader))

prev_time = time.time()
with torch.no_grad():
    for i, batch in enumerate(dataloader):
        # Set model input
        lr = Variable(batch["lr"].type(Tensor))
        min = -10956 # Got from MinMax Normalization of HR images
        max = 6362   # Got from MinMax Normalization of HR images
        mean = Variable(batch["mean"].type(Tensor))
        var = Variable(batch["var"].type(Tensor))
        input_path = batch["input_path"]
        img_root, img_name = os.path.split(input_path[0])

        # ------------------
        #  Generating
        # ------------------
        output = generator(lr)
        #save PNG#
        save_image(output, os.path.join(opt.saved_path, "gen_{}.png".format(img_name)), normalize=True)

        output = (output * var + mean) * (max-min) + min
        output = output.detach().cpu().numpy()
        output = np.float32(output)
#        print(output.shape)

        # save TIF

        print(output[0][0].shape)

        with rasterio.Env():
            with rasterio.open(input_path[0]) as src: 
                profile = src.meta
                print(profile)
                t = src.transform*src.transform.scale(1/4, 1/4)
                
                profile.update(
                    width = opt.img_width * 4, 
                    height = opt.img_height * 4,
                    dtype = rasterio.float32,
                    transform = t 
                )
                print(profile)
            with rasterio.open(os.path.join(opt.saved_path, "gen_{}.tif".format(img_name)), 'w', **profile) as dst:
                dst.write(output[0][0], indexes = 1)