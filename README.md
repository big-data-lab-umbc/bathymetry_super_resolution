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
  - Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
  - PyTorch-GAN Github Repository

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
