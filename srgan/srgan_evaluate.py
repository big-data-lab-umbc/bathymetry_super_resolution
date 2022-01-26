"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 srgan.py'
"""

import argparse
import time

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets import *
from models import *

parser = argparse.ArgumentParser()
#parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
#parser.add_argument("--n_epochs", type=int, default=2, help="number of epochs of training")
#parser.add_argument("--dataset_name", type=str, default="img_hr", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
#parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
#parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
#parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
#parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
#parser.add_argument("--hr_height", type=int, default=512, help="high res. image height")
#parser.add_argument("--hr_width", type=int, default=512, help="high res. image width")
parser.add_argument("--img_height", type=int, default=540, help="size of input image height")
parser.add_argument("--img_width", type=int, default=540, help="size of input image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
#parser.add_argument("--sample_interval", type=int, default=5, help="interval between saving image samples")
#parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
parser.add_argument("--source_path", type=str, default="/home/ubuntu/PyTorch-GAN/implementations/srgan/test_data", help="path of the source data")
parser.add_argument("--saved_path", type=str, default="/home/ubuntu/PyTorch-GAN/implementations/srgan/images", help="path of the generated data")
parser.add_argument("--model", type=str, default="/home/ubuntu/PyTorch-GAN/implementations/srgan/saved_models/tif_noGAN_ep3/generator_99.pth", help="path of the generator")


opt = parser.parse_args()

##JL
#lr_tif = "Tile_33.68_119.9_EPSG4326_Etopo.tif" #"input.tif" #"Tile_33.68_119.9_EPSG4326_Etopo.tif"
#hr_tif = "Hr_"+lr_tif

opt.img_height = 128 #2160
opt.img_width = 128 #2160 
#opt.model = 'weighted_loss_weight=0.1.pth'
#opt.model = 'saved_models/baseline_model.pth'
opt.model = 'saved_models/minmax_scaler_generator_1.pth'
opt.source_path = '../data/img_lr_png_test/'
opt.saved_path = '../data/img_hr_png_test/'


##JL

print(opt)

cuda = torch.cuda.is_available()
print('Cuda available ', cuda)
input_shape = (opt.img_height, opt.img_width)


# Initialize generator and discriminator
generator = GeneratorResNet(in_channels=opt.channels, out_channels=opt.channels)
#discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
feature_extractor = FeatureExtractor()

# Set feature extractor to inference mode
#feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()
if cuda:
    generator = generator.cuda()
#    discriminator = discriminator.cuda()
#    feature_extractor = feature_extractor.cuda()
#    criterion_GAN = criterion_GAN.cuda()
#    criterion_content = criterion_content.cuda()

generator.load_state_dict(torch.load(opt.model))
#discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth"))
print('Load generator successfully')

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# img_test = rasterio.open(self.files_lr[index % len(self.files_lr)])

img_shape = (opt.img_height, opt.img_width)


dataloader = DataLoader(
    #ImageDataset("../../data/%s" % opt.dataset_name, hr_shape=hr_shape),
    ImageDatasetPNGTest(opt.source_path, hr_shape = img_shape),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=0,
)

print('Generating')
print(len(dataloader))

prev_time = time.time()
with torch.no_grad():
    for i, batch in enumerate(dataloader):
        # Set model input
        lr = Variable(batch["lr"].type(Tensor))
        image_name = batch["name"]
        print(batch["name"])

        # ------------------
        #  Generating
        # ------------------
        output = generator(lr)
        print(output.shape)
        print(type(output))

        input_path = os.path.join(opt.source_path, batch["name"][0])
        output_path = os.path.join(opt.saved_path, "Hr_"+batch["name"][0])

        #save PNG#
        save_image(output, os.path.join(output_path), normalize=True)
        #output = output.detach().cpu().numpy()
#       print(output.shape)


        
