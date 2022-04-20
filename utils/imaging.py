# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: Some functions to process the data, for example, which will help us make selections
# Project: One/Few-Shot Learning for Galaxy Zoo

import os
import glob
from typing import Tuple, NewType
import PIL
import numpy as np

PILimage = NewType('PILimage', classmethod)


def object_name(iauname: str) -> Tuple[str, str]:
    """The data in Mike's folder is organised by folder name and image name. The format is such that
    the first few letters correspond to the folder name and the the rest as the image name.

    Args:
        iauname (str): name of image in the csv file containing the tags

    Returns:
        Tuple[str, str]: folder name and file name
    """

    # the folder name is just the first four letters in the image name
    folder = iauname[0:4]

    return folder, iauname + '.png'


def load_image(data_dir: str, filename: str) -> Tuple[PILimage, np.ndarray]:
    """Load the image in both PIL and numpy format. We might want to use the numpy array only.

    Args:
        data_dir (str): the full path where the image is located
        filename (str): the name of the image

    Returns:
        Tuple[PIL.PngImagePlugin.PngImageFile, np.ndarray]: PIL image with all descriptions, numpy array of the image
    """

    # image in PIL format
    im_pil = PIL.Image.open(data_dir + filename).convert("RGB")

    # image as a numpy array
    im_arr = np.asarray(im_pil)

    return im_pil, im_arr


def load_image_full(fname: str) -> Tuple[PILimage, np.ndarray]:
    """Load the image in both PIL and numpy format given a full path.

    Args:
        fname (str): the full path to the file.

    Returns:
        Tuple[PILimage, np.ndarray]: PIL image with all descriptions, numpy array of the image
    """
    # image in PIL format
    im_pil = PIL.Image.open(fname).convert("RGB")

    # image as a numpy array
    im_arr = np.asarray(im_pil)

    return im_pil, im_arr


def compute_statistics(data_dir: str) -> Tuple[list, list]:
    """Given a directory, compute the mean and standard deviation of the images.

    Args:
        data_dir (str): the full path where the images are located

    Returns:
        Tuple[np.ndarray, np.ndarray]: mean and standard deviation of the images
    """

    # get the list of files in the directory
    image_paths = glob.glob(os.path.join(data_dir, "*/*.png"))

    # total number of images
    nimages = len(image_paths)

    # create an empty list to store the images
    images = []

    for i in range(nimages):
        images.append(load_image_full(image_paths[i])[1])

    images = np.asarray(images)

    # compute the statistics (mean and standard deviation)
    mean = np.mean(images, axis=(0, 1, 2))
    std = np.std(images, axis=(0, 1, 2))

    return mean.tolist(), std.tolist()
