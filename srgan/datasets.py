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


#Dataloader for training 
class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        # self.files_hr = sorted(glob.glob(root + "img_hr_tif/*.*"))
        # self.files_lr = sorted(glob.glob(root + "img_lr_tif/*.*"))  
        self.files_mask = sorted(glob.glob(root + "mask_tif/*.*"))
        self.files_hr = sorted(glob.glob(root + "img_hr_normalized/*.*"))
        self.files_lr = sorted(glob.glob(root + "img_lr_normalized/*.*"))  
        self.files_original_hr = sorted(glob.glob(root + "img_hr_tif/*.*"))


    
    def __getitem__(self, index):
        img_hr = rasterio.open(self.files_hr[index % len(self.files_hr)])
        img_lr = rasterio.open(self.files_lr[index % len(self.files_lr)])
        mask = rasterio.open(self.files_mask[index % len(self.files_mask)])

        original_hr = rasterio.open(self.files_original_hr[index % len(self.files_original_hr)])

        input_path = self.files_lr[index % len(self.files_lr)]
        #print(self.files_hr[index % len(self.files_hr)])

        img_hr = np.float32(img_hr.read(1))
        img_lr = np.float32(img_lr.read(1))
        mask = mask.read(1)

        original_hr = np.float32(original_hr.read(1))

        #Transform mask to tensor, and eliminate pixel values of 0 and -inf.
        mask1 = np.where(mask == 1, 1, 0)
        mask2 = np.where(mask == 0, 1, 0)
        masks = mask1 + mask2


        # MinMax + Standard Normalization
        max = np.max([np.max(img_hr),np.max(img_lr)])
        min = np.min([np.min(img_hr),np.min(img_lr)])

        # img_hr_scl = (img_hr-min)/(max-min)
        # img_lr_scl = (img_lr-min)/(max-min)

        # mean = np.mean(img_lr_scl)
        # var = np.var(img_lr_scl)
        mean = 0.5
        var = 0.5

        # image_hr = (img_hr_scl - mean) / var
        # image_lr = (img_lr_scl - mean) / var

        #Transform hr_image and lr_image to tensor
        # image_lr = self.lr_transform(image_lr)
        # image_hr = self.hr_transform(image_hr)
        image_lr = self.lr_transform(img_lr)
        image_hr = self.hr_transform(img_hr)



        def CreatWeight(img_hr, mask):
            #Create an array of weight according to a mask
            weight = (1./img_hr) * mask
            return(np.abs(weight))

        weight = CreatWeight(original_hr, masks)

        return {"lr": image_lr, "hr": image_hr, "weight": weight, "mask": masks, "min": min, "max": max, "mean": mean, "var": var, "input_path": input_path}

    def __len__(self):
        return len(self.files_hr)

class ImageDatasetTiff(Dataset):

    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                # transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                # transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files_hr = sorted(glob.glob(root + "img_hr_normalized/*.*"))
        self.files_lr = sorted(glob.glob(root + "img_lr_normalized/*.*"))   # for training

    def __getitem__(self, index):
        ###        img_hr = Image.open(self.files_hr[index % len(self.files_hr)]).convert('L')
        ###        img_lr = Image.open(self.files_lr[index % len(self.files_lr)]).convert('L')
        img_hr = rasterio.open(self.files_hr[index % len(self.files_hr)])
        img_lr = rasterio.open(self.files_lr[index % len(self.files_lr)])
        # print(self.files_lr[index % len(self.files_lr)])
        # print(self.files_hr[index % len(self.files_hr)])

        input_path = self.files_lr[index % len(self.files_lr)]

        img_hr = img_hr.read(1)*0.5+0.5
        #print(img_hr.shape)
        #print(img_hr)
        img_hr_png = img_hr*255
        #print(img_hr)
        img_hr_png=img_hr_png.astype(np.uint8)

        img_lr = img_lr.read(1)*0.5+0.5
        img_lr_png = img_lr*255
        img_lr_png=img_lr_png.astype(np.uint8)

        image_lr = self.lr_transform(img_lr_png)
        image_hr = self.hr_transform(img_hr_png)


        return {"lr": image_lr, "hr": image_hr, "input_path": input_path}

    def __len__(self):
        return len(self.files_hr)


#Dataloader for testing
class ImageDataset_test(Dataset):
    def __init__(self, root, img_shape):
        img_height, img_width = img_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        
        self.files_lr = sorted(glob.glob(root + "/Tile_26.49_56.58_EPSG4326_Etopo.tif"))
        self.files_hr = sorted(glob.glob(root + "/Tile_26.49_56.58_EPSG4326_Gebco.tif"))    

    def __getitem__(self, index):

        img_lr = rasterio.open(self.files_lr[index % len(self.files_lr)])
        img_hr = rasterio.open(self.files_hr[index % len(self.files_hr)])

        img_lr = np.float32(img_lr.read(1))
        img_hr = np.float32(img_hr.read(1))

        max = np.max([np.max(img_hr),np.max(img_lr)])
        min = np.min([np.min(img_hr),np.min(img_lr)])

        img_lr_scl = (img_lr-min)/(max-min)
        img_hr_scl = (img_hr-min)/(max-min)

        mean = 0.5
        var = 0.5

        image_lr = (img_lr_scl - mean) / var
        image_hr = (img_hr_scl - mean) / var

        image_lr = self.lr_transform(img_lr)
        image_hr = self.hr_transform(image_hr)

        return {"lr": image_lr, "min": min, "max": max, "mean": mean, "var": var}

    def __len__(self):
        return len(self.files_lr)


class ImageDatasetPNGTest(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files_lr = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img_lr = Image.open(self.files_lr[index % len(self.files_lr)])
        image_lr = self.lr_transform(img_lr)

        return {"lr": image_lr, "name": "test.png"}

    def __len__(self):
        return len(self.files_lr)
        image_lr = self.lr_transform(img_lr)


class ImageDatasetPrePNG(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)

class ImageDatasetPretrainTiff(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                # transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                # transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ]
        )

        self.files_hr = sorted(glob.glob(root + "/img_hr_normalized/*.*"))
        self.files_lr = sorted(glob.glob(root + "/img_lr_normalized/*.*"))   

    def __getitem__(self, index):
        img_hr = rasterio.open(self.files_hr[index % len(self.files_hr)])
        img_lr = rasterio.open(self.files_lr[index % len(self.files_lr)])

        input_path = self.files_lr[index % len(self.files_lr)]

        # Read the MinMax-normalized data ranging in [0,1]
        img_hr = img_hr.read(1)
        img_lr = img_lr.read(1)

        # Standard normalization to [-1,1]
        mean = 0.5
        var = 0.5
        img_hr = (img_hr - mean) / var
        img_lr = (img_lr - mean) / var

        # Transfer to 3d array to apply pretrained model
        img_hr = img_hr[:,:,np.newaxis]
        img_hr_3d = np.concatenate((img_hr,img_hr,img_hr),axis=2)
        img_lr = img_lr[:,:,np.newaxis]
        img_lr_3d = np.concatenate((img_lr,img_lr,img_lr),axis=2)

        # Transfer to Tensor data
        image_lr = self.lr_transform(img_lr_3d)
        image_hr = self.hr_transform(img_hr_3d)


        return {"lr": image_lr, "hr": image_hr, "input_path": input_path}

    def __len__(self):
        return len(self.files_hr)

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