# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: This file contain the main settings for running the codes.
# Project: One/Few-Shot Learning for Galaxy Zoo

from torchvision import transforms

# Steps
# 1) Correct for the right location of the images and check if the images exist in Mike's folder.
# 2) Generate the description files for spirals and ellipticals based on selection cuts.
# 3) Pick a subset of this data, for example, 2000 out of 8000.
# 4) Copy the data (per category) from Mike's folder to the new folder ($DATA/data/images/spiral/object.jpg).
# 5) Split the data (csv/dataframe) into training, testing and validation sets.
# 6) Copy the training and validation images into the ml/ folder.

# DECaLS (the data is in Mike's directory on ARC cluster)
decals = '/data/phys-zooniverse/chri5177/galaxy_zoo/decals/dr5/png'

# The Zenodo directory - we will need this for the description of each image (volunteers' votes)
zenodo = '/data/phys-zooniverse/phys2286/data/zenodo'

# Data from my folder (will also contains the tags that Mike shared)
data_dir = '/data/phys-zooniverse/phys2286/data'

# ---------------------------------------------------------------------
# the Deep Learning part
new_img_size = [3, 224, 224]

transformation = [
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Resize(300),
    transforms.CenterCrop(224)]

# basic statistics of the images. These are fixed, meaning same transformation should be applied to
# training, validation and test data.

# mean of the whole dataset (this is for 3 channels)
mean_img = [0.485, 0.456, 0.406]  # [26.97003201762193, 25.172733883647798, 24.687282796368816]

# standard deviation of the whole dataset
std_img = [0.229, 0.224, 0.225]  # [27.974221728738513, 25.714420641820155, 24.653711141402653]

normalise = False

# apply normalisation if set in argument
if normalise:
    transformation.append(transforms.Normalize(mean=mean_img, std=std_img))

# training and validation paths
train_path = data_dir + '/ml/train_images/'
val_path = data_dir + '/ml/validate_images/'
# test_path = data_dir + '/ml/test_images/'
test_path = data_dir + '/ml/generalise/'

# ----------------------------------------------------------------------------------
FRAC = 0.75
N_VOTE = 10

condition_spiral = {'has-spiral-arms_yes_fraction': FRAC, 'has-spiral-arms_yes': N_VOTE}
condition_elliptical = {'smooth-or-featured_smooth_fraction': FRAC,
                        'smooth-or-featured_smooth': N_VOTE, 'merging_none_fraction': FRAC}

condition_strong_bar = {'bar_strong': N_VOTE, 'bar_strong_fraction': FRAC}
condition_merger = {'merging_merger': N_VOTE, 'merging_merger_fraction': FRAC}
condition_bulge_round = {'edge-on-bulge_rounded': N_VOTE, 'edge-on-bulge_rounded_fraction': FRAC}
condition_major_disturbance = {'merging_major-disturbance': N_VOTE, 'merging_major-disturbance_fraction': 0.6}
