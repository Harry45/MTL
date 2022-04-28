"""
Description: Dataloader for the Galaxy Zoo (DECaLS) data.
"""

# Author: Arrykrishna Mootoovaloo
# Date: April 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Project: Multi-Task Learning for Galaxy Zoo

import os
from typing import Tuple
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# our scripts and functions
import settings as st
import utils.helpers as hp


class DECaLSDataset(Dataset):
    """Data loader for the DECaLS dataset.

    Args:
        mode (str): train, validate or test
        augment (bool): whether to augment the data or not. Default is False.
    """

    def __init__(self, mode: str, augment: bool = False):

        path = os.path.join(st.DATA_DIR, 'ml')

        if mode == 'train':
            self.desc = hp.load_csv(path, 'train')

            # to remove later
            self.desc = self.desc.iloc[0:100]

        elif mode == 'test':
            self.desc = hp.load_csv(path, 'test')

            # to remove later
            self.desc = self.desc.iloc[0:20]

        else:
            self.desc = hp.load_csv(path, 'validate')

            # to remove later
            self.desc = self.desc.iloc[0:20]

        # transformations
        trans = st.TRANS

        # if we choose to augment, we apply the horizontal flip
        if augment:
            trans.append(transforms.RandomHorizontalFlip(p=0.5))

        # create the transform
        self.transform = transforms.Compose(trans)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and image and its corresponding label.

        Args:
            idx (int): the index of the image to load.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the image and its label.
        """

        # get the image paths for the pair
        image_path = os.path.join(st.DECALS, self.desc['png_loc'].iloc[idx])

        # get the classes for the pair
        label = torch.from_numpy(self.desc.iloc[idx, 2:].values.astype(int))

        # load the image
        image = Image.open(image_path).convert("RGB")

        # transform the images
        if self.transform:
            image = self.transform(image).float()

        return image, label

    def __len__(self) -> int:
        """The number of images in this particular set.

        Returns:
            int: the number of images
        """
        return self.desc.shape[0]
