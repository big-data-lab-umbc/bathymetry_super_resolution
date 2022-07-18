# Bathymetry Super Resolution
## Overview
The Enhanced Bathymetry Super-Resolution model present a data-driven approach to reconstruct the ocean bathymetric data using SR-ResNet based model trained by GEBCO and
ETOPO-1 dataset. We enhance the SR-ResNet by customized loss functions (water loss and coastal loss) and transfer learning to train the model with limited bathymetry data and generate high-spatial-resolution bathymetry that recover accurate detials in coastal area where pixel values are larger than -160. 

## Table of contents
