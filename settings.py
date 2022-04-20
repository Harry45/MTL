"""
Description: This file contain the main settings for running the codes.
"""

# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Project: One/Few-Shot Learning for Galaxy Zoo

from torchvision import transforms

# DECaLS (the data is in Mike's directory on ARC cluster)
DECALS = '/data/phys-zooniverse/chri5177/galaxy_zoo/decals/dr5/png'

# The Zenodo directory - we will need this for the description of each image (volunteers' votes)
ZENODO = '/data/phys-zooniverse/phys2286/data/zenodo'

# Data from my folder (will also contains the tags that Mike shared)
DATA_DIR = '/data/phys-zooniverse/phys2286/data'

# ---------------------------------------------------------------------
# the Deep Learning part
IMG_SIZE = [3, 224, 224]

TRANS = [transforms.ToTensor(),
         transforms.Grayscale(num_output_channels=1),
         transforms.Resize(300),
         transforms.CenterCrop(224)]

# basic statistics of the images. These are fixed, meaning same transformation should be applied to
# training, validation and test data.

# mean of the whole dataset (this is for 3 channels)
MEAN_IMG = [0.485, 0.456, 0.406]

# standard deviation of the whole dataset
STD_IMG = [0.229, 0.224, 0.225]

NORMALISE = False

# apply normalisation if set in argument
if NORMALISE:
    TRANS.append(transforms.Normalize(mean=MEAN_IMG, std=STD_IMG))
