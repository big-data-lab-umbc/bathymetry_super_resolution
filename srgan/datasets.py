import glob
import random
import os
import numpy as np
import rasterio

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
#mean = np.array([0.485, 0.456, 0.406])
#std = np.array([0.229, 0.224, 0.225])
mean = np.array([0.5])
std = np.array([0.5])


#Dataloader for training 
class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                #transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
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
        self.files_lr = sorted(glob.glob(root + "img_lr_tif/*.*"))    #for training

    def __getitem__(self, index):
###        img_hr = Image.open(self.files_hr[index % len(self.files_hr)]).convert('L')
###        img_lr = Image.open(self.files_lr[index % len(self.files_lr)]).convert('L')
        img_hr = rasterio.open(self.files_hr[index % len(self.files_hr)])
        img_lr = rasterio.open(self.files_lr[index % len(self.files_lr)])
        #print(self.files_lr[index % len(self.files_lr)])
        #print(self.files_hr[index % len(self.files_hr)])

        img_hr = img_hr.read(1)
        img_hr=np.float32(img_hr)
        img_lr = img_lr.read(1)
        img_lr=np.float32(img_lr)

        
        image_lr = self.lr_transform(img_lr)
        image_hr = self.hr_transform(img_hr)

        image_hr = (image_hr+2297)/8659
        image_lr = (image_lr+2297)/8659

        return {"lr": image_lr, "hr": image_hr}

    def __len__(self):
        return len(self.files_hr)


# Dataloader for training
class ImageDatasetStdNorm(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                # transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                ###                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                # transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                ###                transforms.Normalize(mean, std),
            ]
        )

        self.files_hr = sorted(glob.glob(root + "img_hr_std_normalized/*.*"))
        self.files_lr = sorted(glob.glob(root + "img_lr_std_normalized/*.*"))  # for training

    def __getitem__(self, index):
        ###        img_hr = Image.open(self.files_hr[index % len(self.files_hr)]).convert('L')
        ###        img_lr = Image.open(self.files_lr[index % len(self.files_lr)]).convert('L')
        img_hr = rasterio.open(self.files_hr[index % len(self.files_hr)])
        img_lr = rasterio.open(self.files_lr[index % len(self.files_lr)])
        img_lr_name = os.path.basename(img_lr.name)
        # print(self.files_lr[index % len(self.files_lr)])
        # print(self.files_hr[index % len(self.files_hr)])

        img_hr = img_hr.read(1)
        img_hr = np.float32(img_hr)
        img_lr = img_lr.read(1)
        img_lr = np.float32(img_lr)

        image_lr = self.lr_transform(img_lr)
        image_hr = self.hr_transform(img_hr)

        return {"lr": image_lr, "hr": image_hr, "name": img_lr_name}

    def __len__(self):
        return len(self.files_hr)

# Dataloader for training
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

        self.files_hr = sorted(glob.glob(root + "/*.*"))
        self.files_lr = sorted(glob.glob(root + "img_lr_std_normalized/*.*"))  # for training

    def __getitem__(self, index):
        ###        img_hr = Image.open(self.files_hr[index % len(self.files_hr)]).convert('L')
        ###        img_lr = Image.open(self.files_lr[index % len(self.files_lr)]).convert('L')
        img_hr = rasterio.open(self.files_hr[index % len(self.files_hr)])
        img_lr = rasterio.open(self.files_lr[index % len(self.files_lr)])
        img_lr_name = os.path.basename(img_lr.name)
        # print(self.files_lr[index % len(self.files_lr)])
        # print(self.files_hr[index % len(self.files_hr)])

        img_hr = img_hr.read(1)
        img_lr = img_lr.read(1)

        image_lr = self.lr_transform(img_lr)
        image_hr = self.hr_transform(img_hr)

        return {"lr": image_lr, "hr": image_hr, "name": img_lr_name}

    def __len__(self):
        return len(self.files_hr)

# Dataloader for training, it reads minmax normalized data, multiple by 255 and uint8 type so regular ToTensor() and Normalize() could be used.
class ImageDatasetPNGTest(Dataset):

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

        self.files_lr = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        ###        img_hr = Image.open(self.files_hr[index % len(self.files_hr)]).convert('L')
        img_lr = Image.open(self.files_lr[index % len(self.files_lr)]).convert('L')
        #img_lr = rasterio.open(self.files_lr[index % len(self.files_lr)])
        # print(self.files_lr[index % len(self.files_lr)])
        # print(self.files_hr[index % len(self.files_hr)])
        image_lr = self.lr_transform(img_lr)

        return {"lr": image_lr, "name": "test.png"}

    def __len__(self):
        return len(self.files_lr)
        image_lr = self.lr_transform(img_lr)


#Dataloader for testing
class ImageDataset_test(Dataset):
    def __init__(self, root, img_shape):
        img_height, img_width = img_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                #transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        
        self.files_lr = sorted(glob.glob(root + "/*.*"))    

    def __getitem__(self, index):

        img_lr = rasterio.open(self.files_lr[index % len(self.files_lr)])
        img_lr_name = os.path.basename(img_lr.name)

        img_lr = img_lr.read(1)
        img_lr=np.float32(img_lr)

        image_lr = self.lr_transform(img_lr)

        #image_lr = (image_lr+2297)/8659

        return {"lr": image_lr, "name": img_lr_name}

    def __len__(self):
        return len(self.files_lr)
