"""
Description: Train the network for the multi-label case.
"""

# Author: Arrykrishna Mootoovaloo
# Date: April 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Project: Multi-Task Learning for Galaxy Zoo

import os
import torch
import torch.nn as nn
from src.dataset import DECaLSDataset
from torch.utils.data import Dataset, DataLoader
from src.network import MultiLabelNet

out_path = './output/'
os.makedirs(out_path, exist_ok=True)

# create the dataloader
train_dataset = DECaLSDataset(mode='train', augment=False)
val_dataset = DECaLSDataset(mode='validate', augment=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(dataset=val_loader, batch_size=4, shuffle=False)

# define the model
model = MultiLabelNet(backbone="resnet18")

# to assign weights to this loss function
loss_func = nn.MultiLabelSoftMarginLoss(weight=st.CLASS_WEIGHTS, reduction='mean')
