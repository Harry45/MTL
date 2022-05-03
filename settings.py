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


# number of classes we have in the decision tree
NCLASS = 34

# ------------------------------------------------------------------------------------------------
# Multi-Label Learning
# ------------------------------------------------------------------------------------------------

# this is the normalised (inverse) weights
CLASS_WEIGHTS = [0.00208189, 0.00436793, 0.0660458, 0.00749026, 0.00219077,
                 0.00472274, 0.00325833, 0.02512607, 0.01181951, 0.00263074,
                 0.07742019, 0.01529738, 0.00415841, 0.00678291, 0.02441088,
                 0.00453844, 0.00268769, 0.00808778, 0.04412375, 0.01750932,
                 0.00580903, 0.00782268, 0.01158968, 0.016774, 0.04669389,
                 0.00731781, 0.0638864, 0.16710421, 0.19994213, 0.01259986,
                 0.00173038, 0.01981222, 0.065145, 0.03902195]

# ------------------------------------------------------------------------------------------------
# Multi-task Learning
# ------------------------------------------------------------------------------------------------

COLUMNS = [
    'Smooth', 'Featured or Disk', 'Artifact', 'Edge On Disk (Yes)', 'Edge On Disk (No)', 'Spiral Arms (Yes)',
    'Spiral Arms (No)', 'Strong Bar', 'Weak Bar', 'No Bar', 'Central Bulge (Dominant)', 'Central Bulge (Large)',
    'Central Bulge (Moderate)', 'Central Bulge (Small)', 'Central Bulge (None)', 'Round', 'In Between', 'Cigar Shaped',
    'Bulge (Boxy)', 'Bulge (None)', 'Bulge (Rounded)', 'Spiral Winding (Tight)', 'Spiral Winding (Medium)',
    'Spiral Winding (Loose)', 'Spiral Arms (1)', 'Spiral Arms (2)', 'Spiral Arms (3)', 'Spiral Arms (4)',
    'Spiral Arms (More Than 4)', 'Spiral Arms (cannot tell)', 'Merging (None)', 'Merging (Minor Disturbance)',
    'Merging (Major Disturbance)', 'Merging (Merger)']

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
