# Bathymetry Training
## Downloading Datasets
(Dataset required to be publicated)

Instrustion on running the script:
1. Download the dataset from the provided link
2. Save training tiles in the folder 'train' to '../../data/'
3. Save test tiles in the folder 'test' to '../../data/'
4. Run the sript using command 'python3 srgan.py

## The directory of model:
- Your local folder
  - bathymetry_super_resolution
    - data
    - bathymetry training
      - images
      - Loss
      - saved_models
      - srresnet.py
      - models.py
      - datasets.py
Please create folders for saving generated images and saving trained models.

## Models and corresponding loss functions:
  - Model 1: content loss
  - Model 2: content loss + water loss
  - Model 3: water loss
  - Model 4: water loss + coastal loss
  - Model 5: coastal loss
