# -*- coding: utf-8 -*-
"""save_normalized_data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zmNrdgaiRTVY-i6a8WUsKnSWMYjwkDso
"""

import rasterio
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pickle import dump
import glob

imgs_dir = '../data/img_hr_tif/'
#imgs_dir = '../data/img_lr_tif/'
scaled_imgs_dir = '../data/img_hr_std_normalized/'
#scaled_imgs_dir = '../data/img_lr_std_normalized/'
scaler_file_name = '/hr_scaler.pkl'

#Sklearn cannot be used because 2D data is required, but we only have 1D data
img_combined = []
#Load all pixle values of hr images to a 1D array, fit standard scaler
for img in glob.glob(imgs_dir+"*.tif"):
    img_io = rasterio.open(img)
    img_io = img_io.read(1)
    img_combined.append(img_io)
img_combined = np.array(img_combined)
print(img_combined.shape)

#apply sklearn Standard Scaler
scaler = StandardScaler()
img_hr_std_scaled_data = scaler.fit_transform(img_combined.reshape(-1,1))
print("scaler:")
print(scaler.get_params())
dump(scaler, open(scaled_imgs_dir + scaler_file_name, 'wb'))
print("saved scaler done")


#save scaled image
for img in glob.glob(imgs_dir+"*.tif"):
    img_io = rasterio.open(img)
    img_io = img_io.read(1)
    img_scl = np.float32(scaler.transform(img_io.flatten().reshape(-1, 1)))
    img_scl = img_scl.reshape(-1, img_io.shape[1])
    with rasterio.Env():
        with rasterio.open(img) as src:
            profile = src.meta
            profile.update(
                dtype = rasterio.float32,
            )
            print(profile)
        print(os.path.basename(img))
        output_path = os.path.join(scaled_imgs_dir, os.path.basename(img))
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(img_scl, indexes = 1)

# User-defined MinMax Normalization for HR images