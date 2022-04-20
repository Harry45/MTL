"""
Description: Main script for the Galaxy Zoo project.
"""

# Author: Arrykrishna Mootoovaloo
# Date: April 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Project: Multi-Task Learning for Galaxy Zoo

import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# our scripts and functions
from src.dataset import DECaLSDataset
import utils.helpers as hp
import src.processing as sp
import settings as st

# create an output directory
OUT_PATH = './output/'
os.makedirs(OUT_PATH, exist_ok=True)

# Set device to CUDA if a CUDA device is available, else CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def splitting(val_size: float = 0.2, test_size: float = 0.2) -> None:
    """Split the data into train, validate and test. It will automatically save
    the files, which will be used later to train the deep learning model.

    Args:
        val_size (float, optional): The validation size. Defaults to 0.2.
        test_size (float, optional): The test size. Defaults to 0.2.
    """

    # load the votes
    dr5 = hp.read_parquet(st.DATA_DIR, 'descriptions/dr5_votes')

    # generate the labels
    labels = sp.generate_labels(dr5, save=True)

    # train, validate, test
    _ = sp.split_data(labels, val_size, test_size, save=True)


def training(batch_size: int = 8, epochs: int = 10, lr: float = 0.001):

    train_dataset = DECaLSDataset(mode='train', augment=True)
    valid_dataset = DECaLSDataset(mode='validate', augment=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)


if __name__ == '__main__':
    splitting(val_size=0.2, test_size=0.2)
