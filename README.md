# Bathymetry Super Resolution

## Table of contents

- [Table of contents](https://github.com/big-data-lab-umbc/bathymetry_super_resolution/blob/main/README.md#table-of-contents)
- [Architecture](https://github.com/big-data-lab-umbc/bathymetry_super_resolution/blob/main/README.md#architecture)
- [Downloading Datasets](https://github.com/big-data-lab-umbc/bathymetry_super_resolution/blob/main/README.md#downloading-datasets)
- [Training and Testing](https://github.com/big-data-lab-umbc/bathymetry_super_resolution/blob/main/README.md#training-and-testing)
  - [Installation](https://github.com/big-data-lab-umbc/bathymetry_super_resolution/blob/main/README.md#installation)
  - [Training](https://github.com/big-data-lab-umbc/bathymetry_super_resolution/blob/main/README.md#training)
    - [Pretraining](https://github.com/big-data-lab-umbc/bathymetry_super_resolution/blob/main/README.md#pre-training)
    - [Bathymetry training](https://github.com/big-data-lab-umbc/bathymetry_super_resolution/blob/main/README.md#bathymetry-training)
  - [Testing](https://github.com/big-data-lab-umbc/bathymetry_super_resolution/blob/main/README.md#testing)
- [Our Results](https://github.com/big-data-lab-umbc/bathymetry_super_resolution/blob/main/README.md#results)
- [Credit](https://github.com/big-data-lab-umbc/bathymetry_super_resolution/blob/main/README.md#credit)
  - [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802v5.pdf)
  - [PyTorch-GAN Github Repository](https://github.com/eriklindernoren/PyTorch-GAN)



## Architecture

The bathymetry super resolution model is based on a pre-trained SR-ResNet model to generate remote-sensing images with higher resolution than the original ones with limited training dataset. First we pretrain a SR-ResNet model by [Large-scale CelebFaces Attributes (CelebA) Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) which contains 202,599 JPEG images. Then we use transfer learning to adapt our bathymetry dataset composed of 2000 Tiff images selected from [ETOPO-1 dataset](https://www.ngdc.noaa.gov/mgg/global/#:~:text=ETOPO1%20is%20a%201%20arc,base%20of%20the%20ice%20sheets).) and [GEBCO dataset](https://www.gebco.net/), with fine tuning on the last layer of the model. Finally the model is used to generate high-resolution images for any Tiff inputs. Both the pretrained model and bathymetry model are developed using python API and libraries of PyTorch. The figure below illustrated the overall architecture of our bathymetry super resolution model.

![archi](https://user-images.githubusercontent.com/90643297/180836496-6aef8550-6966-4667-9607-acdc6edbf7c0.png)

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

  We use transfer learning to gain some knowledge of super resolution from RGB images because we have only limited training samples. To implement the transfer learning based super resolution for Bathymetry data, we use a pre-training and fine-tuning approach. In particular, transfer learning with the help of a large-scale CelebFaces Attributes dataset is first performed for the external learning of backbone structure and mapping.

  Please refer to README in [pre-training folder](https://github.com/big-data-lab-umbc/bathymetry_super_resolution/tree/main/pretraining) for training steps.

- ##### Bathymetry training

  In bathymetry training process, our customized loss functions are combinations of some or all of three parts: content loss, water loss and coastal loss. We have 5 models of different combinations:

  - Model 1: content loss
  - Model 2: content loss + water loss
  - Model 3: water loss
  - Model 4: water loss + coastal loss
  - Model 5: coastal loss

  Please refer to README in [Bathymetry training folder](https://github.com/big-data-lab-umbc/bathymetry_super_resolution/tree/main/bathymetry_training) for training steps.

### Testing 

The saved trained model by bathymetry training can be used to generate high-resolution images. Load the saved .pth file into [generator](https://github.com/big-data-lab-umbc/bathymetry_super_resolution/blob/main/bathymetry_training/generator.ipynb) to test the saved model.

## Results

The test images are evaluated by MSE, PSNR and SSIM on the whole area, ocean area and coastal area, respectively. Please use codes in [evaluation folder](https://github.com/big-data-lab-umbc/bathymetry_super_resolution/tree/main/evaluation).

The image below illustrates the visual comparison among different experiments and ground truth at the same location. From (a) to (j) are original low-resolution image, bi-cubic interpolation (baseline 1), pre-trained model (baseline 2), directly-trained model (baseline 2), model 1 (content loss), model 2 (content + water loss), model 3 (water loss). model 4 (water + coastal loss), model 5 (coastal loss) and original high-resolution image (ground truth). Among all the experiments, model 5 (coastal loss) is the best model we have. The area in white color has positive pixel values and represents land area, the black area represents deep ocean and the area in grey and around land area represents coastal area.

![cb589411047fb6475159e92caba37fc](https://user-images.githubusercontent.com/90643297/180836330-a17a2d85-f1fb-4620-93ce-efcab41f687e.png)

## Credit

Our work is designed to conduct super resolution of bathymetry data based on theories and network architecture proposed by paper of SR-GAN: *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network*. We developed based on a PyTorch version implementation published at Github repository. The detailed information is as below.

### Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

##### Authors

Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi

[[Paper\]](https://arxiv.org/pdf/1609.04802v5.pdf)

### PyTorch-GAN Github Repository

##### Author: ***[eriklindernoren](https://github.com/eriklindernoren/PyTorch-GAN/commits?author=eriklindernoren)*** 

[[Repository\]](https://github.com/eriklindernoren/PyTorch-GAN)

