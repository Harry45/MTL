"""
Description: Functions for the ranking of the images.

Author: Arrykrishna Mootoovaloo
Date: August 2022
Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
Project: One/Few-Shot Learning for Galaxy Zoo
"""

import os
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# our scripts and functions
import settings as st
import utils.helpers as hp
# from src.fewshot import ml_backbone
from src.network import MultiLabelNet, MultiTaskNet
from src.dataset import DECaLSDataset


def mtl_backbone_decoders(modelname: str):
    """Returns the backbone and the individual decoders for each task given the
    name of the model, for example, mtl-models-2022-6-14/resnet_18_multitask_29.pth

    Args:
        modelname (str): name of the model to use
    """

    # full path to the model
    fullpath = os.path.join(st.MODEL_PATH, modelname)

    # load the model
    loaded_model = torch.load(fullpath)

    # create the model again
    model = MultiTaskNet(backbone="resnet18", output_size=st.LABELS_PER_TASK, resnet_task=True)
    model = nn.DataParallel(model, device_ids=[0])
    model.load_state_dict(loaded_model)
    model.to(st.DEVICE)

    # the backbone for multi-task learning
    backbone = nn.Sequential(*list(model.backbone.children()))

    # the different decoders
    decoders = model.decoders

    return backbone, decoders


def embeddings_ml(backbone: MultiLabelNet, data: DECaLSDataset) -> torch.Tensor:
    """Generates the embeddings for the multi-label network.

    Args:
        backbone (MultiLabelNet): the backbone of the network
        data (DECaLSDataset): the dataset to use

    Returns:
        torch.Tensor: the normalized embeddings for the shared backbone
    """

    # 0 because first is the image and 1 is the label
    embeddings = backbone(data[0].view(1, 1, st.IMG_SIZE[1], st.IMG_SIZE[1]).to(st.DEVICE))
    embeddings = embeddings.view(1, -1)
    embeddings = F.normalize(embeddings)

    return embeddings


def embeddings_mtl(backbone: MultiTaskNet, decoders: MultiTaskNet, data: DECaLSDataset) -> Tuple[torch.Tensor, dict]:
    """Generates the embeddings for the multitask network.

    Args:
        backbone (MultiLabelNet): the backbone of the network
        decoders (MultiTaskNet): the decoders of the network
        data (DECaLSDataset): the dataset to use

    Returns:
        Tuple[torch.Tensor, dict]: the normalized embeddings for the shared backbone and the decoders
    """

    # get the shared embedding vector from the multitask network
    shared = backbone(data[0].view(1, 1, st.IMG_SIZE[1], st.IMG_SIZE[1]).to(st.DEVICE))
    shared = shared.view(1, -1)
    shared = F.normalize(shared)

    # create an empty dictionary to store the task-specific embeddings
    dec = {}

    for k in range(st.NUM_TASKS):

        # label for the current task
        task = 'task_' + str(k + 1)

        # get the decoder for the current task
        model_task = nn.Sequential(*list(decoders[task].children())[:-1])

        # get the embedding for the current task
        embeddings_task = model_task(shared.view(1, 1, 1000)).view(1, -1)

        # normalize the embedding
        dec[task] = F.normalize(embeddings_task)

    return shared, dec


def calculate_distance_mtl(backbone: MultiTaskNet, decoders: MultiTaskNet, reference_id: int,
                           loader: torch.utils.data.DataLoader, pnorm: int = 1, save: bool = False) -> pd.DataFrame:
    """Calculates the Lp distance distance between the embedding vector for the
    reference image and the images in the loader.

    Args:
        backbone (MultiTaskNet): the backbone of the network
        decoders (MultiTaskNet): the decoders of the network
        reference_id (int): the index of the reference image
        loader (torch.utils.data.DataLoader): the loader to use
        pnorm (int): the p-norm to use
        save (bool): whether to save the results or not

    Returns:
        pd.DataFrame: a dataframe with all the pairwise distances
    """

    # get the reference data from the dataloader
    data_ref = loader.dataset[reference_id]

    # get the embedding for the reference image
    shared_ref, dec_ref = embeddings_mtl(backbone, decoders, data_ref)

    # create an empty list to store the distances
    rec_distances = list()

    # number of images in the dataloader
    ntest = len(loader.dataset)

    for i in range(ntest):

        # a dictionary to store the distances
        distances = {}

        # extract the embeddings of the using the Deep Learning model
        shared, tasks = embeddings_mtl(backbone, decoders, loader.dataset[i])

        # calculate the distance between the reference and the current image
        # using the shared embeddings
        distances['shared'] = torch.cdist(shared_ref, shared, pnorm).item()

        for numtask in range(st.NUM_TASKS):

            # name of the task
            task = 'task_' + str(numtask + 1)

            # task-specific distance
            distances[task] = torch.cdist(dec_ref[task], tasks[task], pnorm).item()

        rec_distances.append(distances)

    distances = pd.DataFrame(rec_distances)

    if save:
        hp.save_pd_csv(distances, 'results', f'distances_mtl_{reference_id}')

    return distances


def calculate_distance_ml(backbone: MultiLabelNet, reference_id: int, loader: torch.utils.data.DataLoader,
                          pnorm: int = 1, save: bool = False) -> pd.DataFrame:
    """Calculates the Lp distance distance between the embedding vector for the
    reference image and the images in the loader. Applies only to the multilabel
    network.

    Args:
        backbone (MultiTaskNet): the backbone of the network
        reference_id (int): the index of the reference image
        loader (torch.utils.data.DataLoader): the loader to use
        pnorm (int): the p-norm to use
        save (bool): whether to save the results or not

    Returns:
        pd.DataFrame: a dataframe with all the pairwise distances
    """
    # get the reference data from the dataloader
    data_ref = loader.dataset[reference_id]

    # get the embedding for the reference image
    embedding_ref = embeddings_ml(backbone, data_ref)

    # create an empty list to store the distances
    rec_distances = list()

    # number of images in the dataloader
    ntest = len(loader.dataset)

    # create an empty list to record the distances
    rec_distances = list()

    for i in range(ntest):

        # extract the embeddings of the using the Deep Learning model
        embedding = embeddings_ml(backbone, loader.dataset[i])

        # calculate the distance between the reference and the current image
        # using the shared embeddings
        distance = torch.cdist(embedding_ref, embedding, pnorm).item()

        rec_distances.append(distance)

    distances = pd.DataFrame(rec_distances, columns=['distance'])

    if save:
        hp.save_pd_csv(distances, 'results', f'distances_ml_{reference_id}')

    return distances