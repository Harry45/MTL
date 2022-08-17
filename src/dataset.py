"""
Description: Dataloader for the Galaxy Zoo (DECaLS) data.
"""

# Author: Arrykrishna Mootoovaloo
# Date: April 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Project: Multi-Task Learning for Galaxy Zoo

import os
import glob
from typing import Tuple
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# our scripts and functions
import settings as st
import utils.helpers as hp


class FewShotFineTuneData(Dataset):
    """A dataset for the Galaxy Zoo (DECaLS) data with a fixed number of classes
    for the few shot learning part (transductive finetuning).

    Args:
        subset (bool): If True, the dataset will be Subset, else Query.
        nshots (int): The number of shots to be used for the few shot learning.
    """

    def __init__(self, support: bool, nshot: int = 10):

        # support set or query set
        self.support = support

        # number of shots
        self.nshot = nshot

        # get the transformation to be applied to the data
        trans = st.TRANS

        # build the transformation
        self.transform = transforms.Compose(trans)

        if self.support:
            fname = f'support_targets_{str(nshot)}'

        else:
            fname = f'query_targets_{str(nshot)}'

        self.csvfile = hp.load_csv('fewshot', fname)

        # to remove later
        if not support:
            self.csvfile = self.csvfile.iloc[:50]

    def __getitem__(self, index) -> torch.Tensor:

        # choose a row in the csv file
        row = self.csvfile.iloc[index]

        # the target
        target = row['Targets']

        if self.support:
            filepath = f"fewshot/{str(self.nshot)}-shots/{row['Labels']}/{row['Objects']}"

        else:
            filepath = f"fewshot/query/{row['Objects']}"

        # load the image
        image = Image.open(filepath).convert("RGB")

        # transform the images
        if self.transform:
            image = self.transform(image).float()

        return image, torch.LongTensor([target])

    def __len__(self):
        return len(self.csvfile)


class FSdataset(Dataset):
    """A dataset for the Galaxy Zoo (DECaLS) data with a fixed number of classes
    for the few shot learning part.

    Args:
        subset (bool): If True, the dataset will be Subset, else Query.
        objtype (str): The type of object to be used for the few shot learning.
        nshots (int): The number of shots to be used for the few shot learning.
    """

    def __init__(self, support: bool, **kwargs):

        # get the transformation to be applied to the data
        trans = st.TRANS

        # build the transformation
        self.transform = transforms.Compose(trans)

        if support:

            # record the object type
            self.objtype = kwargs.pop('objtype')

            # number of shots
            nshot = kwargs.pop('nshot')

            # get all the file names for that particular object
            self.fnames = glob.glob(f'fewshot/{str(nshot)}-shots/' + self.objtype + '/*')

        else:

            # the files are the query images
            self.fnames = glob.glob(f'fewshot/query/*')

    def __getitem__(self, index) -> torch.Tensor:

        # load the image
        image = Image.open(self.fnames[index]).convert("RGB")

        # transform the images
        if self.transform:
            image = self.transform(image).float()

        return image

    def __len__(self):
        return len(self.fnames)


class DECaLSDataset(Dataset):
    """Data loader for the DECaLS dataset.

    Args:
        mode (str): train, validate or test
        augment (bool): whether to augment the data or not. Default is False.
        multi_task (bool): whether to use multi-task learning or not. Default is False.
    """

    def __init__(self, mode: str, augment: bool = False, multi_task: bool = False):

        path = os.path.join(st.DATA_DIR, "ml")

        if mode == "train":
            self.desc = hp.load_csv(path, "train")
            print(f"The number of training points is {self.desc.shape[0]}")

            # to remove later (this is for short experiments)
            # self.desc = self.desc.iloc[0:500]

        elif mode == "test":
            self.desc = hp.load_csv(path, "test")
            print(f"The number of test points is {self.desc.shape[0]}")

            # to remove later (this is for short experiments)
            # self.desc = self.desc.iloc[0:200]

        else:
            self.desc = hp.load_csv(path, "validate")
            print(f"The number of validation points is {self.desc.shape[0]}")

            # to remove later (this is for short experiments)
            # self.desc = self.desc.iloc[0:200]

        # transformations
        trans = st.TRANS

        # if we choose to augment, we apply the horizontal flip
        if augment:
            trans.append(transforms.RandomHorizontalFlip(p=0.5))

        # create the transform
        self.transform = transforms.Compose(trans)

        # if we choose to use multi-task learning, we add the labels
        self.multi_task = multi_task

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and image and its corresponding label.

        Args:
            idx (int): the index of the image to load.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the image and its label.
        """

        # get the image path
        image_path = os.path.join(st.DECALS, self.desc["png_loc"].iloc[idx])

        # load the image
        image = Image.open(image_path).convert("RGB")

        # transform the images
        if self.transform:
            image = self.transform(image).float()

        # get the labels
        if self.multi_task:
            dummy_labels = self.desc.iloc[idx, 2:]

            label = dict()
            for i in range(st.NUM_TASKS):
                task = dummy_labels[st.LABELS["task_" + str(i + 1)]].values.astype(int)
                label["task_" + str(i + 1)] = torch.from_numpy(task)

        else:
            label = torch.from_numpy(self.desc.iloc[idx, 2:].values.astype(int))

        return image, label

    def __len__(self) -> int:
        """The number of images in this particular set.

        Returns:
            int: the number of images
        """
        return self.desc.shape[0]
