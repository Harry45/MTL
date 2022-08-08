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
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

# our scripts and functions
import settings as st
import utils.helpers as hp
from src.network import MultiLabelNet
from src.dataset import FSdataset


import warnings
warnings.filterwarnings("ignore")


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

    os.makedirs(dirpath, exist_ok=True)

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


def copy_query_images(nshot: int = 10, save: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Copy the query images to the fewshot/query/ directory.

    Args:
        nshot (int): number of shots we are using. Defaults to 10.
        save (bool, optional): Option to save the files. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of two pandas dataframes,
        query and subset.
    """

    dirpath = 'fewshot/query/'

    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)

    os.makedirs(dirpath, exist_ok=True)

    query_dataframe = list()
    subset_dataframe = list()

    for objtype in st.FS_CLASSES:

        # list the images in the main folder
        main = os.listdir(f'fewshot/images/{objtype}/')

        # list the images in the subset folder
        subset = os.listdir(f'fewshot/{str(nshot)}-subsets/{objtype}/')

        # create the list of query objects
        query = list(set(main) ^ set(subset))

        # number of objects
        nquery = len(query)
        nsubset = len(subset)

        # create dataframes to store the labels
        df_query = pd.DataFrame()
        df_query['Objects'] = query
        df_query['Labels'] = [objtype] * nquery

        df_subset = pd.DataFrame()
        df_subset['Objects'] = subset
        df_subset['Labels'] = [objtype] * nsubset

        # get the full paths for the query objects and copy them
        paths = [f'fewshot/images/{objtype}/' + query[i] for i in range(nquery)]

        for i in range(nquery):
            subprocess.run(["cp", paths[i], dirpath], capture_output=True, text=True, check=True)

        # store the different dataframes
        query_dataframe.append(df_query)
        subset_dataframe.append(df_subset)

    query_dataframe = pd.concat(query_dataframe)
    subset_dataframe = pd.concat(subset_dataframe)

    if save:
        hp.save_pd_csv(query_dataframe, 'fewshot', 'query')
        hp.save_pd_csv(subset_dataframe, 'fewshot', 'subset')

    return query_dataframe, subset_dataframe


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

    # number of shots
    nshots = len(dataloaders[col].dataset)

    if save:
        hp.save_pickle(vectors, "fewshot", f"vectors_{str(nshots)}")
        hp.save_pickle(vectors_mean, "fewshot", f"vectors_mean_{str(nshots)}")

    return vectors, vectors_mean


def distance_subset_query(modelname: str, nshot: int, save: bool) -> pd.DataFrame:
    """Calculates the L1 distance (Manhattan or Taxicab) between the query and
    centroid of the subsets.

    Args:
        modelname (str): The model to use to create the embedding vector.
        nshot (int): The number of examples in the subset data.
        save (bool): Whether to save the data or not.

    Returns:
        pd.DataFrame: The dataframe consisting of the true labels and the
        predicted labels.
    """

    # mean vector computed and stored
    vectors_mean = hp.load_pickle('fewshot', f'vectors_mean_{str(nshot)}')

    # the model loaded
    model = ml_backbone(modelname)

    # the dataloader for the query set
    querydata = FSdataset(subset=False)
    queryloader = DataLoader(dataset=querydata, batch_size=1, shuffle=False)
    nquery = len(queryloader.dataset)

    # create an empty list to store the normalised vectors for the subset.
    class_subset = list()

    for key in st.FS_CLASSES:
        v_norm = F.normalize(vectors_mean[key].view(1, -1))
        class_subset.append(v_norm)

    # convert to a tensor (this is of size 4 x 1000)
    class_subset = torch.cat(class_subset, dim=0)

    print(f'The shape of the subsets embeddings is {class_subset.shape[0]} x {class_subset.shape[1]}')

    distance_l1 = dict()

    for idx in range(nquery):
        img = queryloader.dataset[idx].view(1, 1, st.IMG_SIZE[-1], st.IMG_SIZE[-1]).to(st.DEVICE)
        vec = model(img).data.to('cpu').view(-1)

        # normalise vector
        vec_norm = F.normalize(vec.view(1, -1))

        # pairwise distance
        dist = torch.cdist(vec_norm, class_subset, p=1)

        # name of the file
        name = os.path.split(queryloader.dataset.fnames[idx])[-1]
        distance_l1[name] = dist.view(-1).data.numpy()

    distance_l1 = pd.DataFrame(distance_l1).T

    # rename the columns
    distance_l1.columns = st.FS_CLASSES

    # find the column name with the minimum distance
    labels_pred = distance_l1.idxmin(axis="columns")
    labels_pred = labels_pred.reset_index(level=0)

    # rename the columns
    labels_pred.columns = ['Objects', 'Labels']

    # load the true labels and merge them with the predicted labels
    truth = hp.load_csv('fewshot', 'query')
    combined = pd.merge(truth, labels_pred, on='Objects', how='outer')

    # rename the columns for the combined dataframe
    combined.columns = ['Objects', 'True Labels', 'Predicted Labels']

    if save:
        hp.save_pd_csv(combined, 'fewshot', 'truth_predicted')

    return combined
