import glob
import random
import os
import numpy as np
import rasterio
import math
from skimage.transform import rescale

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

        # self.files_mask = sorted(glob.glob(root + "mask_tif/*.*"))
        self.files_hr = sorted(glob.glob(root + "hr/*.*"))
        self.files_lr = sorted(glob.glob(root + "lr/*.*"))  
    
    def __getitem__(self, index):
        img_hr = rasterio.open(self.files_hr[index % len(self.files_hr)])
        img_lr = rasterio.open(self.files_lr[index % len(self.files_lr)])
        # mask = rasterio.open(self.files_mask[index % len(self.files_mask)])

        input_path = self.files_lr[index % len(self.files_lr)]
        #print(self.files_hr[index % len(self.files_hr)])

        img_hr = np.float32(img_hr.read(1))
        img_lr = np.float32(img_lr.read(1))
        # mask = mask.read(1)

        #Transform mask to tensor, and eliminate pixel values of 0 and -inf.
        # mask1 = np.where(mask == 1, 1, 0)
        # mask2 = np.where(mask == 0, 1, 0)
        # masks = mask1 + mask2

        # MinMax + Standard Normalization
        max_hr = 6787
        min_hr = -10802
        max_lr = 6392
        min_lr = -9820

        img_hr_scl = (img_hr-min_hr)/(max_hr-min_hr)
        img_lr_scl = (img_lr-min_lr)/(max_lr-min_lr)

        # mean = np.mean(img_lr_scl)
        # var = np.var(img_lr_scl)
        mean = 0.5
        var = 0.5

        image_hr = (img_hr_scl - mean) / var
        image_lr = (img_lr_scl - mean) / var

        #Transform hr_image and lr_image to tensor
        # image_lr = self.lr_transform(image_lr)
        # image_hr = self.hr_transform(image_hr)
        image_lr = self.lr_transform(image_lr)
        image_hr = self.hr_transform(image_hr)

        # def CreatWeight(img_hr, mask):
        #     #Create an array of weight according to a mask
        #     weight = (1./img_hr) * mask
        #     return(np.abs(weight))
        # weight = CreatWeight(original_hr, masks)

        return {"lr": image_lr, "hr": image_hr, "min": min_hr, "max": max_hr, "mean": mean, "var": var, "input_path": input_path}

    def __len__(self):
        return len(self.files_hr)


#Dataloader for no scale training 
class ImageDatasetNoScale(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        self.hr_height = hr_height
        self.hr_width = hr_width
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

        # self.files_mask = sorted(glob.glob(root + "mask_tif/*.*"))
        self.files_hr = sorted(glob.glob(root + "hr/*.*"))
        self.files_lr = sorted(glob.glob(root + "lr/*.*"))  
    
    def __getitem__(self, index):
        

        img_hr = rasterio.open(self.files_hr[index % len(self.files_hr)])
        img_lr = rasterio.open(self.files_lr[index % len(self.files_lr)])
        # mask = rasterio.open(self.files_mask[index % len(self.files_mask)])

        input_path = self.files_lr[index % len(self.files_lr)]

        img_hr = np.float32(img_hr.read(1))
        img_lr = np.float32(img_lr.read(1))
        img_hr = img_hr[:self.hr_height, :self.hr_width]
        img_lr = img_lr[:int(self.hr_height/4), :int(self.hr_width/4)]
        img_lr =  rescale(img_lr, 4, anti_aliasing=False, order=3)

        # MinMax + Standard Normalization
        # Parameters are calculated in another script
        # Script: https://github.com/big-data-lab-umbc/bathymetry_super_resolution/blob/main/data-preprocessing/save_normalized_data.ipynb
        max_hr = 6787
        min_hr = -10802
        max_lr = 6392
        min_lr = -9820

        img_hr_scl = (img_hr-min_hr)/(max_hr-min_hr)
        img_lr_scl = (img_lr-min_lr)/(max_lr-min_lr)

        mean = 0.5
        var = 0.5

        image_hr = (img_hr_scl - mean) / var
        image_lr = (img_lr_scl - mean) / var

        #Transform hr_image and lr_image to tensor
        image_lr = self.lr_transform(image_lr)
        image_hr = self.hr_transform(image_hr)

        return {"lr": image_lr, "hr": image_hr, "min": min_hr, "max": max_hr, "mean": mean, "var": var, "input_path": input_path}

    def __len__(self):
        return len(self.files_hr)



# Data loader for transferring TIF data to PNG format (8uint)
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
        
        self.files_lr = sorted(glob.glob(root + "/*.*"))
        self.files_hr = sorted(glob.glob(root + "/*.*"))    

    def __getitem__(self, index):

        img_lr = rasterio.open(self.files_lr[index % len(self.files_lr)])
        img_hr = rasterio.open(self.files_hr[index % len(self.files_hr)])

        input_path = self.files_lr[index % len(self.files_lr)]

        img_lr = np.float32(img_lr.read(1))
        img_hr = np.float32(img_hr.read(1))

        min = -9820
        max = 6392

        img_lr_scl = (img_lr-min)/(max-min)
        img_hr_scl = (img_hr-min)/(max-min)

        mean = 0.5
        var = 0.5

        image_lr = (img_lr_scl - mean) / var
        image_hr = (img_hr_scl - mean) / var

        image_lr = self.lr_transform(img_lr)
        image_hr = self.hr_transform(image_hr)

        return {"lr": image_lr, "min": min, "max": max, "mean": mean, "var": var, "input_path" :input_path}

    def __len__(self):
        return len(self.files_lr)

# Data loader for testing PNG images
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

# Data loader for training PNG dataset with a pre-trained model
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

# Data loader for training TIF dataset with a pre-trained model
class ImageDatasetPretrainTiff(Dataset):
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

        self.files_hr = sorted(glob.glob(root + "hr/*.*"))
        self.files_lr = sorted(glob.glob(root + "lr/*.*"))   

    def __getitem__(self, index):
        img_hr = rasterio.open(self.files_hr[index % len(self.files_hr)])
        img_lr = rasterio.open(self.files_lr[index % len(self.files_lr)])

        input_path = self.files_lr[index % len(self.files_lr)]

        # Read the MinMax-normalized data ranging in [0,1]
        img_hr = img_hr.read(1)
        img_lr = img_lr.read(1)

        max_lr = 6392
        min_lr = -9820
        max_hr = 6787
        min_hr = -10802
        img_hr_scl = (img_hr-min_hr)/(max_hr-min_hr)
        img_lr_scl = (img_lr-min_lr)/(max_lr-min_lr)

        # Standard normalization to [-1,1]
        mean = 0.5
        var = 0.5
        img_hr = (img_hr_scl - mean) / var
        img_lr = (img_lr_scl - mean) / var

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

# Data loader for training TIF dataset with a pre-trained model
class Truncate(Dataset):
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

        self.files_hr = sorted(glob.glob(root + "hr/*.*"))
        self.files_lr = sorted(glob.glob(root + "lr/*.*"))   

    def __getitem__(self, index):
        img_hr = rasterio.open(self.files_hr[index % len(self.files_hr)])
        img_lr = rasterio.open(self.files_lr[index % len(self.files_lr)])

        input_path = self.files_lr[index % len(self.files_lr)]

        # Read the MinMax-normalized data ranging in [0,1]
        img_hr = img_hr.read(1)
        img_lr = img_lr.read(1)
        img_hr = np.where(img_hr>=0,0,img_hr)
        img_hr = np.where(img_hr<=-255, -255, img_hr)
        img_lr = np.where(img_lr>=0,0,img_lr)
        img_lr = np.where(img_lr<=-255,-255, img_lr)

        max_lr = 0
        min_lr = -255
        max_hr = 0
        min_hr = -255
        img_hr_scl = (img_hr-min_hr)/(max_hr-min_hr)
        img_lr_scl = (img_lr-min_lr)/(max_lr-min_lr)

        # Standard normalization to [-1,1]
        mean = 0.5
        var = 0.5
        img_hr = (img_hr_scl - mean) / var
        img_lr = (img_lr_scl - mean) / var

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

# Data loader for training TIF dataset with a pre-trained model
class Sigmoid(Dataset):
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

        self.files_hr = sorted(glob.glob(root + "hr/*.*"))
        self.files_lr = sorted(glob.glob(root + "lr/*.*"))   

    def __getitem__(self, index):
        img_hr = rasterio.open(self.files_hr[index % len(self.files_hr)])
        img_lr = rasterio.open(self.files_lr[index % len(self.files_lr)])

        input_path = self.files_lr[index % len(self.files_lr)]

        # Read the MinMax-normalized data ranging in [0,1]
        img_hr = img_hr.read(1)
        img_lr = img_lr.read(1)

        img_hr_scl = 1/(1+np.exp(img_hr*(-1)))
        img_lr_scl = 1/(1+np.exp(img_lr*(-1)))

        # Standard normalization to [-1,1]
        mean = 0.5
        var = 0.5
        img_hr = (img_hr_scl - mean) / var
        img_lr = (img_lr_scl - mean) / var

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


# Data loader for training TIF dataset with a pre-trained model
class ImageDatasetPretrainMask(Dataset):
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

        self.files_hr = sorted(glob.glob(root + "hr/*.*"))
        self.files_lr = sorted(glob.glob(root + "lr/*.*"))
        self.files_mask = sorted(glob.glob(root + "mask_hr/*.*"))   

    def __getitem__(self, index):
        img_hr = rasterio.open(self.files_hr[index % len(self.files_hr)])
        img_lr = rasterio.open(self.files_lr[index % len(self.files_lr)])
        mask = rasterio.open(self.files_mask[index % len(self.files_mask)])

        input_path = self.files_lr[index % len(self.files_lr)]

        # Read the data as numpy
        img_hr = img_hr.read(1)
        img_lr = img_lr.read(1)
        mask = mask.read(1)

        mask1 = np.where(mask == 1, 1, 0)
        mask2 = np.where(mask == 0, 1, 0)
        mask = mask1 + mask2

        #Create weight
        def CreatWeight(img_hr, mask):
            #Create an array of weight according to a mask
            weight = (1./img_hr) * mask
            return(np.abs(weight))
        weight = CreatWeight(img_hr, mask)

        #Min-max normalization to [0,1]
        max_hr = 6787
        min_hr = -10802
        max_lr = 6392
        min_lr = -9820
        img_hr_scl = (img_hr-min_hr)/(max_hr-min_hr)
        img_lr_scl = (img_lr-min_lr)/(max_lr-min_lr)

        # mask_hr = img_hr_scl * mask

        # Standard normalization to [-1,1]
        mean = 0.5
        var = 0.5
        img_hr_scl = (img_hr_scl - mean) / var
        img_lr_scl = (img_lr_scl - mean) / var

        # Transfer to 3d array to apply pretrained model
        img_hr_scl = img_hr_scl[:,:,np.newaxis]
        img_hr_3d = np.concatenate((img_hr_scl,img_hr_scl,img_hr_scl),axis=2)
        img_lr_scl = img_lr_scl[:,:,np.newaxis]
        img_lr_3d = np.concatenate((img_lr_scl,img_lr_scl,img_lr_scl),axis=2)
        mask = mask[:,:,np.newaxis]
        mask_3d = np.concatenate((mask,mask,mask),axis=2)
        weight = weight[:,:,np.newaxis]
        weight_3d = np.concatenate((weight,weight,weight),axis=2)

        # Transfer to Tensor data
        image_lr = self.lr_transform(img_lr_3d)
        image_hr = self.hr_transform(img_hr_3d)
        masks = self.hr_transform(mask_3d) 
        weights = self.hr_transform(weight_3d)
        # image_hr_mask = self.hr_transform(mask_hr_3d)

        return {"lr": image_lr, "hr": image_hr, "input_path": input_path, "weight": weights, "mask": masks}

    def __len__(self):
        return len(self.files_hr)

# Data loader for training TIF dataset with a pre-trained model
class TruncateMask(Dataset):
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

        self.files_hr = sorted(glob.glob(root + "hr/*.*"))
        self.files_lr = sorted(glob.glob(root + "lr/*.*"))
        self.files_mask = sorted(glob.glob(root + "mask_hr/*.*"))   

    def __getitem__(self, index):
        img_hr = rasterio.open(self.files_hr[index % len(self.files_hr)])
        img_lr = rasterio.open(self.files_lr[index % len(self.files_lr)])
        mask = rasterio.open(self.files_mask[index % len(self.files_mask)])

        input_path = self.files_lr[index % len(self.files_lr)]

        # Read the data as numpy
        img_hr = img_hr.read(1)
        img_lr = img_lr.read(1)
        mask = mask.read(1)

        mask1 = np.where(mask == 1, 1, 0)
        mask2 = np.where(mask == 0, 1, 0)
        mask = mask1 + mask2

        #Create weight
        def CreatWeight(img_hr, mask):
            #Create an array of weight according to a mask
            weight = (1./img_hr) * mask
            return(np.abs(weight))
        weight = CreatWeight(img_hr, mask)

        # Truncate the input image
        img_hr = np.where(img_hr>=0,0,img_hr)
        img_hr = np.where(img_hr<=-255,-255, img_hr)
        img_lr = np.where(img_lr>=0,0,img_lr)
        img_lr = np.where(img_lr<=-255,-255, img_lr)

        # Min-max normalization to [0,1]
        max_hr = 0
        min_hr = -255
        max_lr = 0
        min_lr = -255
        img_hr_scl = (img_hr-min_hr)/(max_hr-min_hr)
        img_lr_scl = (img_lr-min_lr)/(max_lr-min_lr)

        # mask_hr = img_hr_scl * mask

        # Standard normalization to [-1,1]
        mean = 0.5
        var = 0.5
        img_hr_scl = (img_hr_scl - mean) / var
        img_lr_scl = (img_lr_scl - mean) / var

        # Transfer to 3d array to apply pretrained model
        img_hr_scl = img_hr_scl[:,:,np.newaxis]
        img_hr_3d = np.concatenate((img_hr_scl,img_hr_scl,img_hr_scl),axis=2)
        img_lr_scl = img_lr_scl[:,:,np.newaxis]
        img_lr_3d = np.concatenate((img_lr_scl,img_lr_scl,img_lr_scl),axis=2)
        mask = mask[:,:,np.newaxis]
        mask_3d = np.concatenate((mask,mask,mask),axis=2)
        weight = weight[:,:,np.newaxis]
        weight_3d = np.concatenate((weight,weight,weight),axis=2)

        # Transfer to Tensor data
        image_lr = self.lr_transform(img_lr_3d)
        image_hr = self.hr_transform(img_hr_3d)
        masks = self.hr_transform(mask_3d) 
        weights = self.hr_transform(weight_3d)
        # image_hr_mask = self.hr_transform(mask_hr_3d)

        return {"lr": image_lr, "hr": image_hr, "input_path": input_path, "weight": weights, "mask": masks}

    def __len__(self):
        return len(self.files_hr)

# Data loader for training TIF dataset with a pre-trained model
class SigmoidMask(Dataset):
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

        self.files_hr = sorted(glob.glob(root + "hr/*.*"))
        self.files_lr = sorted(glob.glob(root + "lr/*.*"))
        self.files_mask = sorted(glob.glob(root + "mask_hr/*.*"))   

    def __getitem__(self, index):
        img_hr = rasterio.open(self.files_hr[index % len(self.files_hr)])
        img_lr = rasterio.open(self.files_lr[index % len(self.files_lr)])
        mask = rasterio.open(self.files_mask[index % len(self.files_mask)])

        input_path = self.files_lr[index % len(self.files_lr)]

        # Read the data as numpy
        img_hr = img_hr.read(1)
        img_lr = img_lr.read(1)
        mask = mask.read(1)

        mask1 = np.where(mask == 1, 1, 0)
        mask2 = np.where(mask == 0, 1, 0)
        mask = mask1 + mask2

        #Create weight
        def CreatWeight(img_hr, mask):
            #Create an array of weight according to a mask
            weight = (1./img_hr) * mask
            return(np.abs(weight))
        weight = CreatWeight(img_hr, mask)

        #Min-max normalization to [0,1]
        img_hr_scl = 1/(1+np.exp(img_hr*(-1)))
        img_lr_scl = 1/(1+np.exp(img_lr*(-1)))

        # Standard normalization to [-1,1]
        mean = 0.5
        var = 0.5
        img_hr_scl = (img_hr_scl - mean) / var
        img_lr_scl = (img_lr_scl - mean) / var

        # Transfer to 3d array to apply pretrained model
        img_hr_scl = img_hr_scl[:,:,np.newaxis]
        img_hr_3d = np.concatenate((img_hr_scl,img_hr_scl,img_hr_scl),axis=2)
        img_lr_scl = img_lr_scl[:,:,np.newaxis]
        img_lr_3d = np.concatenate((img_lr_scl,img_lr_scl,img_lr_scl),axis=2)
        mask = mask[:,:,np.newaxis]
        mask_3d = np.concatenate((mask,mask,mask),axis=2)
        weight = weight[:,:,np.newaxis]
        weight_3d = np.concatenate((weight,weight,weight),axis=2)

        # Transfer to Tensor data
        image_lr = self.lr_transform(img_lr_3d)
        image_hr = self.hr_transform(img_hr_3d)
        masks = self.hr_transform(mask_3d) 
        weights = self.hr_transform(weight_3d)
        # image_hr_mask = self.hr_transform(mask_hr_3d)

        return {"lr": image_lr, "hr": image_hr, "input_path": input_path, "weight": weights, "mask": masks}

    def __len__(self):
        return len(self.files_hr)

class Weight_1000_Mask(Dataset):
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

        self.files_hr = sorted(glob.glob(root + "hr/*.*"))
        self.files_lr = sorted(glob.glob(root + "lr/*.*"))
        self.files_mask = sorted(glob.glob(root + "mask_hr/*.*"))   

    def __getitem__(self, index):
        img_hr = rasterio.open(self.files_hr[index % len(self.files_hr)])
        img_lr = rasterio.open(self.files_lr[index % len(self.files_lr)])
        mask = rasterio.open(self.files_mask[index % len(self.files_mask)])

        input_path = self.files_lr[index % len(self.files_lr)]

        # Read the data as numpy
        img_hr = img_hr.read(1)
        img_lr = img_lr.read(1)
        mask = mask.read(1)

        mask1 = np.where(mask == 1, 1, 0)
        mask2 = np.where(mask == 0, 1, 0)
        mask = mask1 + mask2

        #Create weight
        def CreatWeight(img_hr, mask):
            #Create an array of weight according to a mask
            weight = (1000./img_hr) * mask
            return(np.abs(weight))
        weight = CreatWeight(img_hr, mask)

        #Min-max normalization to [0,1]
        max_hr = 6787
        min_hr = -10802
        max_lr = 6392
        min_lr = -9820
        img_hr_scl = (img_hr-min_hr)/(max_hr-min_hr)
        img_lr_scl = (img_lr-min_lr)/(max_lr-min_lr)

        # mask_hr = img_hr_scl * mask

        # Standard normalization to [-1,1]
        mean = 0.5
        var = 0.5
        img_hr_scl = (img_hr_scl - mean) / var
        img_lr_scl = (img_lr_scl - mean) / var

        # Transfer to 3d array to apply pretrained model
        img_hr_scl = img_hr_scl[:,:,np.newaxis]
        img_hr_3d = np.concatenate((img_hr_scl,img_hr_scl,img_hr_scl),axis=2)
        img_lr_scl = img_lr_scl[:,:,np.newaxis]
        img_lr_3d = np.concatenate((img_lr_scl,img_lr_scl,img_lr_scl),axis=2)
        mask = mask[:,:,np.newaxis]
        mask_3d = np.concatenate((mask,mask,mask),axis=2)
        weight = weight[:,:,np.newaxis]
        weight_3d = np.concatenate((weight,weight,weight),axis=2)

        # Transfer to Tensor data
        image_lr = self.lr_transform(img_lr_3d)
        image_hr = self.hr_transform(img_hr_3d)
        masks = self.hr_transform(mask_3d) 
        weights = self.hr_transform(weight_3d)
        # image_hr_mask = self.hr_transform(mask_hr_3d)

        return {"lr": image_lr, "hr": image_hr, "input_path": input_path, "weight": weights, "mask": masks}

    def __len__(self):
        return len(self.files_hr)

# Data loader for testing single TIF image with a pre-trained model
class ImageDatasetPretrainTest(Dataset):
    def __init__(self, root, hr_shape):
        img_height, img_width = hr_shape
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
        min = -9820
        max = 6392
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

# Data loader for testing single TIF image with a pre-trained model
class TruncateTest(Dataset):
    def __init__(self, root, hr_shape):
        img_height, img_width = hr_shape
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
        img_lr = np.where(img_lr>=0,0,img_lr)
        img_lr = np.where(img_lr<=-255,-255, img_lr)

        # MinMax normalization to [0,1];
        min = -255
        max = 0
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


# Data loader for training TIF dataset with a pre-trained model
class ImageDatasetBaseline2_Water(Dataset):
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

        self.files_hr = sorted(glob.glob(root + "hr/*.*"))
        self.files_lr = sorted(glob.glob(root + "lr/*.*"))
        self.files_maskhr = sorted(glob.glob(root + "mask_hr/*.*"))  
        self.files_masklr = sorted(glob.glob(root + "mask_lr/*.*"))  

    def __getitem__(self, index):
        img_hr = rasterio.open(self.files_hr[index % len(self.files_hr)])
        img_lr = rasterio.open(self.files_lr[index % len(self.files_lr)])
        mask_hr = rasterio.open(self.files_maskhr[index % len(self.files_maskhr)])
        mask_lr = rasterio.open(self.files_masklr[index % len(self.files_masklr)])

        input_path = self.files_lr[index % len(self.files_lr)]

        # Read the data as numpy
        img_hr = np.float32(img_hr.read(1))
        img_lr = np.float32(img_lr.read(1))
        mask_hr = mask_hr.read(1)
        mask_lr = mask_lr.read(1)

        mask1 = np.where(mask_hr == 1, 1, 0)
        mask2 = np.where(mask_hr == 0, 1, 0)
        mask_hr = mask1 + mask2
        mask3 = np.where(mask_lr == 1, 1, 0)
        mask4 = np.where(mask_lr == 0, 1, 0)
        mask_lr = mask3 + mask4

        #Create weight
        def CreatWeight(img_hr, mask):
            #Create an array of weight according to a mask
            weight = (1./img_hr) * mask
            return(np.abs(weight))
        weight = CreatWeight(img_hr, mask_hr)

        # Only do Min-max normalization on Water area to [0,1]
        max_hr = 6787
        min_hr = -10802
        max_lr = 6392
        min_lr = -9820
    
        img_hr_scl = (img_hr-min_hr)/(max_hr-min_hr)
        img_lr_scl = (img_lr-min_lr)/(max_lr-min_lr)

        # mask_hr = img_hr_scl * mask

        # Standard normalization to [-1,1]
        mean = 0.5
        var = 0.5
        img_hr_scl = (img_hr_scl - mean) / var
        img_lr_scl = (img_lr_scl - mean) / var

        img_hr_water = img_hr_scl * mask_hr
        img_lr_water = img_lr_scl * mask_lr

        # Transfer to 3d array to apply pretrained model
        img_hr_water = img_hr_water[:,:,np.newaxis]
        img_hr_3d = np.concatenate((img_hr_water,img_hr_water,img_hr_water),axis=2)

        img_lr_water = img_lr_water[:,:,np.newaxis]
        img_lr_3d = np.concatenate((img_lr_water,img_lr_water,img_lr_water),axis=2)

        mask_hr = mask_hr[:,:,np.newaxis]
        mask_hr_3d = np.concatenate((mask_hr,mask_hr,mask_hr),axis=2)

        weight = weight[:,:,np.newaxis]
        weight_3d = np.concatenate((weight,weight,weight),axis=2)

        # Transfer to Tensor data
        image_lr = self.lr_transform(img_lr_3d)
        image_hr = self.hr_transform(img_hr_3d)
        masks = self.hr_transform(mask_hr_3d) 
        weights = self.hr_transform(weight_3d)
        # image_hr_mask = self.hr_transform(mask_hr_3d)

        return {"lr": image_lr, "hr": image_hr, "input_path": input_path, "weight": weights, "mask": masks}

    def __len__(self):
        return len(self.files_hr)

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