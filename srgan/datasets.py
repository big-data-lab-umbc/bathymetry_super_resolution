import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
#mean = np.array([0.485, 0.456, 0.406])
#std = np.array([0.229, 0.224, 0.225])
mean = np.array([0.5,])
std = np.array([0.5,])


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                #transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                #transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files_hr = sorted(glob.glob(root + "img_hr/*.*"))
        self.files_lr = sorted(glob.glob(root + "img_lr/*.*"))

    def __getitem__(self, index):
        img_hr = Image.open(self.files_hr[index % len(self.files_hr)]).convert('L')
        img_lr = Image.open(self.files_lr[index % len(self.files_lr)]).convert('L')
        #print(self.files_lr[index % len(self.files_lr)])
        #print(self.files_hr[index % len(self.files_hr)])
        image_lr = self.lr_transform(img_lr)
        image_hr = self.hr_transform(img_hr)

        return {"lr": image_lr, "hr": image_hr}

    def __len__(self):
        return len(self.files_hr)
