"""
Description: Networks related to the few shot learning part.

Author: Arrykrishna Mootoovaloo
Date: August 2022
Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
Project: One/Few-Shot Learning for Galaxy Zoo
"""

import os
import shutil
import random
import subprocess
import torch
import torch.nn as nn
import pandas as pd

# our scripts and functions
import settings as st
import utils.helpers as hp
from src.network import MultiLabelNet


def select_objects(dataframe: pd.DataFrame, columns: list,
                   threshold: float = 0.90, nobjects: int = 20, save: bool = False) -> dict:
    """Select objects from the pandas dataframe given a threshold for the
    volunteers' votes

    Args:
        dataframe (pd.DataFrame): a dataframe with the volunteers votes
        columns (list): a list of the columns, for which the threshold will be
        applied
        threshold (float, optional): the threshold value. Defaults to 0.90.
        nobjects (int, optional): the number of objects to be selected. Defaults to 20.
        save (bool, optional): Option to save the files. Defaults to False.

    Returns:
        dict: Dictionary of the selected objects
    """
    dataframe.rename(st.MAPPING, axis=1, inplace=True)

    objects = dict()

    for col in columns:
        objects[col] = dataframe[dataframe[col] > threshold]

        print(f'{col} has {len(objects[col])} objects')

        assert len(objects[col]) > nobjects, "too many objects selected"

        randint = random.sample(range(0, len(objects[col])), nobjects)

        objects[col] = objects[col].iloc[randint]

        # reset pandas index
        objects[col].reset_index(drop=True, inplace=True)

    if save:
        hp.save_pickle(objects, 'fewshot', 'attributes')

    return objects


def copy_image_fewshot(nobjects: int = 50, threshold: float = 0.90):
    """Copy examples of similar objects from Mike's folder to the fewshot/images/ directory.

    Args:
        nobjects (int): number of objects to copy. Defaults to 50.
        threshold (float): the threshold to apply to the volunteers' votes. Defaults to 0.90.
    """

    # remove existing images if they exist
    dirpath = "fewshot/images/"
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)

    # read the dataframe
    dataframe = hp.read_parquet(st.DATA_DIR, 'descriptions/dr5_votes')

    # we need to show the network images that it has never seen before (robust analysis)
    testpoints = hp.load_csv(st.DATA_DIR, 'ml/test')
    subdata = dataframe[dataframe.iauname.isin(testpoints.iauname)]

    # select the objects
    objects = select_objects(subdata, st.FS_COLS, threshold, nobjects=nobjects, save=True)

    for col in st.FS_COLS:

        # select the common objects
        obj = objects[col]

        # name of the folder
        colname = col.replace(" ", "-").replace("(", "").replace(")", "")

        # folder where we want to save the objects
        folder = 'fewshot/images/' + colname

        # make the folder
        os.makedirs(folder, exist_ok=True)

        for i in range(nobjects):
            path = st.DECALS + '/' + obj.png_loc.iloc[i]
            subprocess.run(["cp", path, folder], capture_output=True, text=True)


def ml_backbone(modelname: str):
    """Returns the model (backbone) which outputs the embeddings for the
    image.This function is for the multilabel case only.

    Args:
        modelname (str): name of the trained model, for example,
        ml-models-2022-5-25/resnet_18_multilabel_29.pth
    """

    # full path to the model
    fullpath = os.path.join(st.MODEL_PATH, modelname)

    # load the model
    loaded_model = torch.load(fullpath)

    # create the model again
    model = MultiLabelNet(backbone="resnet18")
    model = nn.DataParallel(model, device_ids=[0])
    model.load_state_dict(loaded_model)
    model.to(st.DEVICE)

    # the backbone
    chopped_layer = nn.Sequential(list(model.children())[0].backbone)

    return chopped_layer


def ml_feature_extractor(model: torch.nn.modules, dataloaders: dict, save: bool) -> torch.Tensor:
    """Extract the embeddings from the trained model given an image.

    Args:
        model (torch.nn.modules): the backbone.
        dataloaders (dict): the dataloaders for the support set.
        save (bool): whether to save the embeddings or not.

    Returns:
        torch.Tensor: the feature vector
    """

    # create an empty dictionary to store the features
    vectors = dict()

    # an empty list to store the mean embedding for each class
    vectors_mean = dict()

    for col in st.FS_CLASSES:

        # get the dataloader for the current class
        loader = dataloaders[col]

        # create an empty list to store the features for the current class
        vectors[col] = list()

        for i in range(st.NSHOTS):
            img = loader.dataset[i].view(1, 1, st.IMG_SIZE[-1], st.IMG_SIZE[-1]).to(st.DEVICE)
            vec = model(img).data.to('cpu').view(-1)
            vectors[col].append(vec)

        # convert the list to a tensor (10 x 1000) - each row is the embedding for a single image
        vectors[col] = torch.vstack(vectors[col])
        vectors_mean[col] = vectors[col].mean(dim=0)

    if save:
        hp.save_pickle(vectors, "fewshot", "vectors")
        hp.save_pickle(vectors_mean, "fewshot", "vectors_mean")

    return vectors, vectors_mean
