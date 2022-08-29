"""
Description: Clustering part in order to generate the buckets.

Author: Arrykrishna Mootoovaloo
Date: August 2022
Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
Project: One/Few-Shot Learning for Galaxy Zoo
"""

import os
import shutil
import subprocess
import torch
import pandas as pd
from kmeans_pytorch import kmeans
from torch.utils.data import DataLoader

# our script and functions
import utils.helpers as hp
import settings as st


def cluster_embeddings(model: torch.nn.modules, dataloader: DataLoader, save: bool) -> torch.Tensor:
    """Calculates the embeddings given a model (backbone) and a dataloader.

    Args:
        model (torch.nn.modules): the model (backbone).
        dataloader (DataLoader): the data loader.
        save (bool): option to save the vectors

    Returns:
        torch.Tensor: a tensor of size: Nobjects x 1000.
    """

    record_vectors = list()

    for i in range(st.NOBJECTS_CLUSTERS):

        # the image in the right format
        img = dataloader.dataset[i][0].view(1, 1, st.IMG_SIZE[-1], st.IMG_SIZE[-1]).to(st.DEVICE)

        # the embedding vector
        vec = model(img).data.to('cpu').view(-1)

        # a list to store the vector
        record_vectors.append(vec)

    record_vectors = torch.vstack(record_vectors)

    # save the embedding vectors
    if save:
        hp.save_pickle(record_vectors, "clustering", "embedding_vectors")

    return record_vectors


def cluster_kmeans(dataloader: DataLoader, vectors: torch.Tensor, save: bool) -> pd.DataFrame:
    """Apply k-means clustering on the embedding vectors.

    Args:
        dataloader (DataLoader): the file with the descriptions.
        vectors (torch.Tensor): the embedding vectors.
        save (bool): option to save the outputs.

    Returns:
        pd.DataFrame: a pandas dataframe with the last column giving the cluster
        ID.
    """

    # k-means clustering
    ids, _ = kmeans(X=vectors, num_clusters=st.NUM_CLUSTERS, distance='euclidean', device=st.DEVICE)

    # description of the objects
    df_subset = dataloader.dataset.desc.iloc[0:st.NOBJECTS_CLUSTERS]
    df_cluster = pd.DataFrame(ids, columns=['Cluster ID'])
    combined = pd.concat([df_subset, df_cluster], axis=1)

    if save:
        hp.save_pd_csv(combined, 'clustering', 'clusters')

    return combined


def cluster_copy_images(dataframe: pd.DataFrame):
    """Copy images from Mike's folder to the clustering folder.

    Args:
        dataframe (pd.DataFrame): a dataframe with the cluster ID
    """

    for i in range(st.NUM_CLUSTERS):

        # subset of the dataframe
        subset = dataframe[dataframe['Cluster ID'] == i]
        nobjects = subset.shape[0]

        # check if folder already exists
        dirpath = 'clustering/images/class_' + str(i)
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)

        os.makedirs(dirpath, exist_ok=True)

        # copy images
        for j in range(nobjects):
            path = os.path.join(st.DECALS, subset['png_loc'].values[j])
            subprocess.run(["cp", path, dirpath], capture_output=True, text=True)


def name_to_id(loader: DataLoader, name: str) -> int:
    """Get the index ID from the dataframe, given the object's name.

    Args:
        loader (DataLoader): the dataloader.
        name (str): name of the object.

    Returns:
        int: the index in the dataframe.
    """

    iaunames = loader.dataset.desc['iauname'].values
    nobjects = len(iaunames)
    exists = [iaunames[i].startswith(name) for i in range(nobjects)]
    index = loader.dataset.desc[exists].index.values[0]
    return index
