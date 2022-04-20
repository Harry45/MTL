# Author: Arrykrishna Mootoovaloo
# Date: April 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: Dataloader for the Galaxy Zoo (DECaLS) data.
# Project: Multi-Task Learning for Galaxy Zoo

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# our scripts and functions
import settings as st
import utils.helpers as hp


class DECaLSDataset(Dataset):

    def __init__(self, mode: str, augment: bool):

        path = os.path.join(st.data_dir, 'ml')

        if mode == 'train':
            self.desc = hp.load_csv(path, 'train')

        elif mode == 'test':
            self.desc = hp.load_csv(path, 'test')

        else:
            self.desc = hp.load_csv(path, 'validate')

        # transformations
        trans = st.transformation

        # if we choose to augment, we apply the horizontal flip
        if augment:

            trans.append(transforms.RandomHorizontalFlip(p=0.5))

        # create the transform
        self.transform = transforms.Compose(trans)

    def __getitem__(self, idx: int):

        # get the image paths for the pair
        image_path = os.path.join(st.decals, self.desc['png_loc'].iloc[idx])

        # get the classes for the pair
        label = torch.from_numpy(self.desc.iloc[idx, 2:].values)

        # load the image
        image = Image.open(image_path).convert("RGB")

        # transform the images
        if self.transform:
            image = self.transform(image).float()

        return image, label

    def __len__(self):
        return self.desc.shape[0]
