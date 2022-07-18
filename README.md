# Bathymetry Super Resolution
## Overview
The Enhanced Bathymetry Super-Resolution model present a data-driven approach to reconstruct the ocean bathymetric data using SR-ResNet based model trained by GEBCO and
ETOPO-1 dataset. We enhance the SR-ResNet by customized loss functions (water loss and coastal loss) and transfer learning to train the model with limited bathymetry data and generate high-spatial-resolution bathymetry that recover accurate detials in coastal area where pixel values are larger than -160. 

## Table of contents
- Overview
- Table of contents
- Architecture
- How to Train and test
  - Training
    - Pretraining
    - Bathymetry training
  - Testing
- Results
- Credit
  - [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802v5.pdf)
  - [PyTorch-GAN Github Repository](https://github.com/eriklindernoren/PyTorch-GAN)

## Architecture

The bathymetry super resolution model is based on a pre-trained SR-ResNet model to generate remote-sensing images with higher resolution than the original ones with limited training dataset. First we pretrain a SR-ResNet model by [Large-scale CelebFaces Attributes (CelebA) Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) which contains 202,599 JPEG images. Then we use transfer learning to adapt our bathymetry dataset composed of 2000 Tiff images selected from [ETOPO-1 dataset](https://www.ngdc.noaa.gov/mgg/global/#:~:text=ETOPO1%20is%20a%201%20arc,base%20of%20the%20ice%20sheets).) and [GEBCO dataset](https://www.gebco.net/), with fine tuning on the last layer of the model. Finally the model is used to generate high-resolution images for any Tiff inputs. Both the pretrained model and bathymetry model are developed using python API and libraries of PyTorch. The figure below illustrated the overall architecture of our bathymetry super resolution model.
![archi](https://user-images.githubusercontent.com/90643297/179571170-0a1b3c47-41fb-407d-9c08-3bfa0e720c7b.png)

## Credit
### Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

*Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi*

[[Paper\]](https://arxiv.org/pdf/1609.04802v5.pdf)

```
@InProceedings{srgan,
    author = {Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi},
    title = {Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network},
    booktitle = {arXiv},
    year = {2016}
}
```

Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

### PyTorch-GAN Github Repository

*[eriklindernoren](https://github.com/eriklindernoren/PyTorch-GAN/commits?author=eriklindernoren)* 

[[Repository\]](https://github.com/eriklindernoren/PyTorch-GAN)
