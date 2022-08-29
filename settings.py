"""
Description: This file contain the main settings for running the codes.

Author: Arrykrishna Mootoovaloo
Date: January 2022
Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
Project: One/Few-Shot Learning for Galaxy Zoo
"""

import numpy as np
import torch
from torchvision import transforms

# device to use
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DECaLS (the data is in Mike's directory on ARC cluster)
DECALS = '/data/phys-zooniverse/chri5177/galaxy_zoo/decals/dr5/png'

# The Zenodo directory - we will need this for the description of each image (volunteers' votes)
ZENODO = '/data/phys-zooniverse/phys2286/data/zenodo'

# Data from my folder
DATA_DIR = '/data/phys-zooniverse/phys2286/data'

# Path where the models are stored

MODEL_PATH = 'models/'  # '/data/phys-zooniverse/phys2286/Models/'

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


# number of classes we have in the decision tree
NCLASS = 34

# ------------------------------------------------------------------------------------------------
# Multi-Label Learning
# ------------------------------------------------------------------------------------------------

# this is the normalised (inverse) weights
CLASS_WEIGHTS = [0.00282107, 0.00568958, 0.05749345, 0.00643104, 0.00381596,
                 0.01139869, 0.01070528, 0.00313111, 0.02441808, 0.04193426,
                 0.01797117, 0.00225253, 0.00815922, 0.05766884, 0.02382508,
                 0.00364199, 0.01460745, 0.02924271, 0.00674986, 0.00465689,
                 0.01009907, 0.01396181, 0.02051897, 0.05561343, 0.00950056,
                 0.06109528, 0.14755674, 0.18717117, 0.01508213, 0.02792996,
                 0.00851766, 0.00534751, 0.01816175, 0.08282971]

# ------------------------------------------------------------------------------------------------
# Multi-task Learning
# ------------------------------------------------------------------------------------------------

# COLUMNS = [
#     'Smooth', 'Featured or Disk', 'Artifact', 'Edge On Disk (Yes)', 'Edge On Disk (No)', 'Spiral Arms (Yes)',
#     'Spiral Arms (No)', 'Strong Bar', 'Weak Bar', 'No Bar', 'Central Bulge (Dominant)', 'Central Bulge (Large)',
#     'Central Bulge (Moderate)', 'Central Bulge (Small)', 'Central Bulge (None)', 'Round', 'In Between',
#     'Cigar Shaped', 'Bulge (Boxy)', 'Bulge (None)', 'Bulge (Rounded)', 'Spiral Winding (Tight)',
#     'Spiral Winding (Medium)', 'Spiral Winding (Loose)', 'Spiral Arms (1)', 'Spiral Arms (2)', 'Spiral Arms (3)',
#     'Spiral Arms (4)', 'Spiral Arms (More Than 4)', 'Spiral Arms (cannot tell)', 'Merging (None)',
#     'Merging (Minor Disturbance)', 'Merging (Major Disturbance)', 'Merging (Merger)']

# Smooth                         0.002821
# Featured or Disk               0.005690
# Artifact                       0.057493
# Round                          0.006431
# In Between                     0.003816
# Cigar Shaped                   0.011399
# Edge On Disk(Yes)             0.010705
# Edge On Disk(No)              0.003131
# Merging(Merger)               0.024418
# Merging(Major Disturbance)    0.041934
# Merging(Minor Disturbance)    0.017971
# Merging(None)                 0.002253
# Bulge(Rounded)                0.008159
# Bulge(Boxy)                   0.057669
# Bulge(None)                   0.023825
# No Bar                         0.003642
# Weak Bar                       0.014607
# Strong Bar                     0.029243
# Spiral Arms(Yes)              0.006750
# Spiral Arms(No)               0.004657
# Spiral Winding(Tight)         0.010099
# Spiral Winding(Medium)        0.013962
# Spiral Winding(Loose)         0.020519
# Spiral Arms(1)                0.055613
# Spiral Arms(2)                0.009501
# Spiral Arms(3)                0.061095
# Spiral Arms(4)                0.147557
# Spiral Arms(More Than 4)      0.187171
# Spiral Arms(cannot tell)      0.015082
# Central Bulge(None)           0.027930
# Central Bulge(Small)          0.008518
# Central Bulge(Moderate)       0.005348
# Central Bulge(Large)          0.018162
# Central Bulge(Dominant)       0.082830

# Models trained
# weighted: ml-models-2022-5-25
# weighted: mtl-models-2022-6-14
# unweighted: mtl-models-2022-6-2

WEIGHTS_MTL = {'task_1': torch.tensor([0.002821, 0.005690, 0.057493]),
               'task_2': torch.tensor([0.006431, 0.003816, 0.011399]),
               'task_3': torch.tensor([0.010705, 0.003131]),
               'task_4': torch.tensor([0.024418, 0.041934, 0.017971, 0.002253]),
               'task_5': torch.tensor([0.008159, 0.057669, 0.023825]),
               'task_6': torch.tensor([0.003642, 0.014607, 0.029243]),
               'task_7': torch.tensor([0.006750, 0.004657]),
               'task_8': torch.tensor([0.010099, 0.013962, 0.020519]),
               'task_9': torch.tensor([0.055613, 0.009501, 0.061095, 0.147557, 0.187171, 0.015082]),
               'task_10': torch.tensor([0.027930, 0.008518, 0.005348, 0.018162, 0.082830])
               }

LABELS = {

    'task_1': ['Smooth', 'Featured or Disk', 'Artifact'],
    'task_2': ['Round', 'In Between', 'Cigar Shaped'],
    'task_3': ['Edge On Disk (Yes)', 'Edge On Disk (No)'],
    'task_4': ['Merging (Merger)', 'Merging (Major Disturbance)', 'Merging (Minor Disturbance)', 'Merging (None)'],
    'task_5': ['Bulge (Rounded)', 'Bulge (Boxy)', 'Bulge (None)'],
    'task_6': ['No Bar', 'Weak Bar', 'Strong Bar'],
    'task_7': ['Spiral Arms (Yes)', 'Spiral Arms (No)'],
    'task_8': ['Spiral Winding (Tight)', 'Spiral Winding (Medium)', 'Spiral Winding (Loose)'],
    'task_9': ['Spiral Arms (1)', 'Spiral Arms (2)', 'Spiral Arms (3)', 'Spiral Arms (4)',
               'Spiral Arms (More Than 4)', 'Spiral Arms (cannot tell)'],
    'task_10': ['Central Bulge (None)', 'Central Bulge (Small)', 'Central Bulge (Moderate)',
                'Central Bulge (Large)', 'Central Bulge (Dominant)']

}

NUM_TASKS = 10

LABELS_PER_TASK = {'task_1': 3,
                   'task_2': 3,
                   'task_3': 2,
                   'task_4': 4,
                   'task_5': 3,
                   'task_6': 3,
                   'task_7': 2,
                   'task_8': 3,
                   'task_9': 6,
                   'task_10': 5
                   }

TASKS_ORDERED = np.concatenate([LABELS['task_' + str(i + 1)] for i in range(NUM_TASKS)])

MAPPING = {'smooth-or-featured_smooth_fraction': 'Smooth',
           'smooth-or-featured_featured-or-disk_fraction': 'Featured or Disk',
           'smooth-or-featured_artifact_fraction': 'Artifact',
           'disk-edge-on_yes_fraction': 'Edge On Disk (Yes)',
           'disk-edge-on_no_fraction': 'Edge On Disk (No)',
           'has-spiral-arms_yes_fraction': 'Spiral Arms (Yes)',
           'has-spiral-arms_no_fraction': 'Spiral Arms (No)',
           'bar_strong_fraction': 'Strong Bar',
           'bar_weak_fraction': 'Weak Bar',
           'bar_no_fraction': 'No Bar',
           'bulge-size_dominant_fraction': 'Central Bulge (Dominant)',
           'bulge-size_large_fraction': 'Central Bulge (Large)',
           'bulge-size_moderate_fraction': 'Central Bulge (Moderate)',
           'bulge-size_small_fraction': 'Central Bulge (Small)',
           'bulge-size_none_fraction': 'Central Bulge (None)',
           'how-rounded_round_fraction': 'Round',
           'how-rounded_in-between_fraction': 'In Between',
           'how-rounded_cigar-shaped_fraction': 'Cigar Shaped',
           'edge-on-bulge_boxy_fraction': 'Bulge (Boxy)',
           'edge-on-bulge_none_fraction': 'Bulge (None)',
           'edge-on-bulge_rounded_fraction': 'Bulge (Rounded)',
           'spiral-winding_tight_fraction': 'Spiral Winding (Tight)',
           'spiral-winding_medium_fraction': 'Spiral Winding (Medium)',
           'spiral-winding_loose_fraction': 'Spiral Winding (Loose)',
           'spiral-arm-count_1_fraction': 'Spiral Arms (1)',
           'spiral-arm-count_2_fraction': 'Spiral Arms (2)',
           'spiral-arm-count_3_fraction': 'Spiral Arms (3)',
           'spiral-arm-count_4_fraction': 'Spiral Arms (4)',
           'spiral-arm-count_more-than-4_fraction': 'Spiral Arms (More Than 4)',
           'spiral-arm-count_cant-tell_fraction': 'Spiral Arms (cannot tell)',
           'merging_none_fraction': 'Merging (None)',
           'merging_minor-disturbance_fraction': 'Merging (Minor Disturbance)',
           'merging_major-disturbance_fraction': 'Merging (Major Disturbance)',
           'merging_merger_fraction': 'Merging (Merger)'}

# ------------------------------------------------------------------------------------------------
# Few Shot Learning
# ------------------------------------------------------------------------------------------------

# the columns to work with in the few shot learning dataset
FS_COLS = ['Artifact', 'Cigar Shaped', 'Merging (Merger)', 'Spiral Arms (Yes)']

# the renamed classes (just for the folders' names)
FS_CLASSES = ['Artifact', 'Cigar-Shaped', 'Merging-Merger', 'Spiral-Arms-Yes']

# number of ways (classes)
NWAYS = len(FS_CLASSES)

# number of examples per class in the few shot learning dataset
NSHOTS = 10


# ------------------------------------------------------------------------------------------------
# Clustering
# ------------------------------------------------------------------------------------------------

# number of clusters we want to use
NUM_CLUSTERS = 10

# Number of objects to apply clustering to
NOBJECTS_CLUSTERS = 1000
