import glob
import random
import os
from cv2 import normalize
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import rasterio

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])
mean = np.array([0.5,])
std = np.array([0.5,])



# Data loader for testing single TIF image with a pre-trained model
class ImageDatasetPretrainTest(Dataset):
    def __init__(self, root, img_shape):
        img_height, img_width = img_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.files_lr = sorted(glob.glob(root + "/*.*"))   

    def __getitem__(self, index):
        img_lr = rasterio.open(self.files_lr[index % len(self.files_lr)])

        input_path = self.files_lr[index % len(self.files_lr)]

        # Read the original tif data
        img_lr = np.float32(img_lr.read(1))

        # MinMax normalization to [0,1]; for lr data: min = -10898, max = 6151
        min = -10898
        max = 6151
        img_lr = (img_lr-min)/(max-min)

        # Standard normalization to [-1,1]
        mean = 0.5
        var = 0.5
        img_lr = (img_lr - mean) / var

        # Transfer to 3d array to apply pretrained model
        img_lr = img_lr[:,:,np.newaxis]
        img_lr_3d = np.concatenate((img_lr,img_lr,img_lr),axis=2)

        # Transfer to Tensor data
        image_lr = self.lr_transform(img_lr_3d)


        return {"lr": image_lr, "input_path": input_path, "mean": mean, "var": var}

    def __len__(self):
        return len(self.files_lr)


# Dataloader for output masks and see whether they put into the model properly
# Not for training or testing
class ImageDataset_mask(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.ToTensor(),
###                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                #transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
###                transforms.Normalize(mean, std),
            ]
        )

        self.files_hr = sorted(glob.glob(root + "img_hr_tif/*.*"))
        self.files_lr = sorted(glob.glob(root + "img_lr_tif/*.*"))  
        self.files_mask = sorted(glob.glob(root + "mask_tif/*.*")) 

    
    def __getitem__(self, index):
###        img_hr = Image.open(self.files_hr[index % len(self.files_hr)]).convert('L')
###        img_lr = Image.open(self.files_lr[index % len(self.files_lr)]).convert('L')
        img_hr = rasterio.open(self.files_hr[index % len(self.files_hr)])
        img_lr = rasterio.open(self.files_lr[index % len(self.files_lr)])
        mask = rasterio.open(self.files_mask[index % len(self.files_mask)])
        #print(self.files_lr[index % len(self.files_lr)])
        #print(self.files_hr[index % len(self.files_hr)])

        img_hr = np.float32(img_hr.read(1))
        img_lr = np.float32(img_lr.read(1))
        mask = mask.read(1)

        # MinMax + Standard Normalization
        max_single = np.max(img_hr)
        max = np.max([np.max(img_hr),np.max(img_lr)])
        min = np.min([np.min(img_hr),np.min(img_lr)])

        img_hr_scl = (img_hr-min)/(max-min)
        img_lr_scl = (img_lr-min)/(max-min)

        # mean = np.mean(img_lr_scl)
        # var = np.var(img_lr_scl)
        mean = 0.5
        var = 0.5

        # image_hr = (img_hr_scl - mean) / var
        # image_lr = (img_lr_scl - mean) / var
        image_hr = (img_hr_scl - mean) / var
        image_lr = (img_lr_scl - mean) / var

        # Generating mask
        mask1 = np.where(mask == 1, 1, 0)
        mask2 = np.where(mask == 0, 1, 0)
        masks = mask1 + mask2
        
        image_lr = self.lr_transform(image_lr)
        image_hr = self.hr_transform(image_hr)

        # image_hr = (image_hr+2297)/8659
        # image_lr = (image_lr+2297)/8659

        def CreatWeight(img_hr, mask):
            #Create an array of weight according to a mask
            weight = (1./img_hr) * mask
            return(np.abs(weight))

        weight = CreatWeight(img_hr, masks)
        recip_img = 1./img_hr

        return {"lr": img_lr_scl, "hr": image_hr, "weight": weight, "recip_img": recip_img, "original_hr": img_hr, "mask": masks, "min":min, "max":max, "mean":mean, "var": var, "max_single": max_single, "mask": masks}

    def __len__(self):
        return len(self.files_hr)