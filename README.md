# Bathymetry Super Resolution

## Table of contents

- Overview
- Table of contents
- Architecture
- Downloading Datasets
- Training and Testing
  - Installation
  - Training
    - Pretraining
    - Bathymetry training
  - Testing
    - Generate results using saved models
  - Evaluation
- Our Results
- Credit
  - [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802v5.pdf)
  - [PyTorch-GAN Github Repository](https://github.com/eriklindernoren/PyTorch-GAN)



## Architecture

The bathymetry super resolution model is based on a pre-trained SR-ResNet model to generate remote-sensing images with higher resolution than the original ones with limited training dataset. First we pretrain a SR-ResNet model by [Large-scale CelebFaces Attributes (CelebA) Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) which contains 202,599 JPEG images. Then we use transfer learning to adapt our bathymetry dataset composed of 2000 Tiff images selected from [ETOPO-1 dataset](https://www.ngdc.noaa.gov/mgg/global/#:~:text=ETOPO1%20is%20a%201%20arc,base%20of%20the%20ice%20sheets).) and [GEBCO dataset](https://www.gebco.net/), with fine tuning on the last layer of the model. Finally the model is used to generate high-resolution images for any Tiff inputs. Both the pretrained model and bathymetry model are developed using python API and libraries of PyTorch. The figure below illustrated the overall architecture of our bathymetry super resolution model.

![archi](https://user-images.githubusercontent.com/90643297/180827702-3fd0b939-b096-45e6-9abc-8b5001274ad9.png)

## Downloading Datasets

- ##### Download Celeb-A datasets for Pre-training:

  The dataset is public and can be downloaded at the [official website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Unzip the folder as *'img_align_celeba'* at the [data folder](https://github.com/big-data-lab-umbc/bathymetry_super_resolution/tree/main/data).

- ##### Download training set for bathymetry training:

  The images for training set of our work are manually cropped from ETOPO-1 and GEBCO grid datasets. - Permission required for public -

- ##### Download test set for testing:

  The images for test set of our work are manually cropped from ETOPO-1 and GEBCO grid datasets. - Permission required for public -

## Training and Testing

### Installation

```
$ git clone https://github.com/big-data-lab-umbc/bathymetry_super_resolution
$ cd bathymetry_super_resolution/
$ sudo pip3 install -r requirements.txt
```

### Training

- ##### Pre-training

  The pre-training process

  Please refer to README in [pre-training folder](https://github.com/big-data-lab-umbc/bathymetry_super_resolution/tree/main/pretraining).

- Bathymetry training

  Please refer to README in [Bathymetry training folder](https://github.com/big-data-lab-umbc/bathymetry_super_resolution/tree/main/bathymetry_training).

### Testing 

Please refer to README in

### Evaluation



## Results



## Credit

Our work is designed to conduct super resolution of bathymetry data based on theories and network architecture proposed by paper of SR-GAN: *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network*. We developed based on a PyTorch version implementation published at Github repository. The detailed information is as below.

### Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

##### Authors

Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi

[[Paper\]](https://arxiv.org/pdf/1609.04802v5.pdf)

### PyTorch-GAN Github Repository

##### Author: ***[eriklindernoren](https://github.com/eriklindernoren/PyTorch-GAN/commits?author=eriklindernoren)*** 

[[Repository\]](https://github.com/eriklindernoren/PyTorch-GAN)

