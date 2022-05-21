"""
Description: This file contain the main settings for running the codes.
"""

# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Project: One/Few-Shot Learning for Galaxy Zoo

import numpy as np
from torchvision import transforms

# DECaLS (the data is in Mike's directory on ARC cluster)
DECALS = '/data/phys-zooniverse/chri5177/galaxy_zoo/decals/dr5/png'

# The Zenodo directory - we will need this for the description of each image (volunteers' votes)
ZENODO = '/data/phys-zooniverse/phys2286/data/zenodo'

# Data from my folder
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
# Multi-task Learning below
# ------------------------------------------------------------------------------------------------

# COLUMNS = [
#     'Smooth', 'Featured or Disk', 'Artifact', 'Edge On Disk (Yes)', 'Edge On Disk (No)', 'Spiral Arms (Yes)',
#     'Spiral Arms (No)', 'Strong Bar', 'Weak Bar', 'No Bar', 'Central Bulge (Dominant)', 'Central Bulge (Large)',
#     'Central Bulge (Moderate)', 'Central Bulge (Small)', 'Central Bulge (None)', 'Round', 'In Between',
#     'Cigar Shaped', 'Bulge (Boxy)', 'Bulge (None)', 'Bulge (Rounded)', 'Spiral Winding (Tight)',
#     'Spiral Winding (Medium)', 'Spiral Winding (Loose)', 'Spiral Arms (1)', 'Spiral Arms (2)', 'Spiral Arms (3)',
#     'Spiral Arms (4)', 'Spiral Arms (More Than 4)', 'Spiral Arms (cannot tell)', 'Merging (None)',
#     'Merging (Minor Disturbance)', 'Merging (Major Disturbance)', 'Merging (Merger)']

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

# tasks to do
# - metrics: take rows where there is only one label per task and calculate the metrics
