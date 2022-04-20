# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: This file contains some helper functions.
# Project: One/Few-Shot Learning for Galaxy Zoo


import pickle
import os
import json
from typing import NewType
import torch
import numpy as np
import pandas as pd

TorchTensor = NewType('TorchTensor', classmethod)


def tensor_to_dict(tensor: TorchTensor, keys: list) -> dict:
    """Convert a tensor to a dictionary given a list of keys

    Args:
        tensor (torch.tensor): The tensor to convert.
        keys (list): The list of keys.

    Returns:
        dict: A dictionary with the keys and values of the tensor.
    """
    return {key: tensor[i].item() for i, key in enumerate(keys)}


def dict_to_tensor(dictionary: dict, keys: list) -> TorchTensor:
    """Converts a dictionary to a tensor.

    Args:
        dictionary (dict): the dictionary to convert
        keys (list): the list of keys (usually in the setting file)

    Returns:
        torch.tensor: the pytorch tensor
    """

    return torch.tensor([dictionary[key] for key in keys])


def subset_dict(dictionary: dict, keys: list) -> dict:
    """Generates a subset of a dictionary.

    Args:
        dictionary (dict): A long dictionary with keys and values respectively.
        keys (list): A list of keys to be extracted.

    Returns:
        dict: A dictionary with only the keys specified.
    """

    return {key: dictionary[key] for key in keys}


def store_arrays(array: np.ndarray, folder_name: str, file_name: str) -> None:
    """Stores a numpy array in a folder.

    Args:
        array (np.ndarray): The array to store.
        folder_name (str): The name of the folder.
        file_name (str): The name of the file.
    """

    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # use compressed format to store data
    np.savez_compressed(folder_name + '/' + file_name + '.npz', array)


def load_arrays(folder_name: str, file_name: str) -> np.ndarray:
    """Load the arrays from a folder.

    Args:
        folder_name (str): name of the folder.
        file_name (str): name of the file.

    Returns:
        np.ndarray: The array.
    """

    matrix = np.load(folder_name + '/' + file_name + '.npz')['arr_0']

    return matrix


def load_csv(folder_name: str, file_name: str) -> pd.DataFrame:
    """Given a folder name and file name, we will load the csv file.

    Args:
        folder_name(str): the name of the folder
        file_name(str): name of the file

    Returns:
        pd.DataFrame: the loaded csv file
    """
    path = folder_name + '/' + file_name + '.csv'

    if not os.path.isfile(path):
        raise FileNotFoundError('File not found: ' + path)

    else:
        df = pd.read_csv(path)
        return df


def save_csv(array: np.ndarray, folder_name: str, file_name: str) -> None:
    """Save an array to a csv file

    Args:
        array (np.ndarray): The array to be saved
        folder_name (str): The name of the folder
        file_name (str): The name of the file
    """
    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    np.savetxt(folder_name + '/' + file_name + '.csv', array, delimiter=',')


def save_pd_csv(df: pd.DataFrame, folder_name: str, file_name: str) -> None:
    """Save an array to a csv file

    Args:
        array (np.ndarray): The array to be saved
        folder_name (str): The name of the folder
        file_name (str): The name of the file
    """
    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    df.to_csv(folder_name + '/' + file_name + '.csv', index=False)


def save_parquet(df: pd.DataFrame, folder_name: str, file_name: str) -> None:
    """Save a dataframe to a parquet file

    Args:
        df(pd.DataFrame): The dataframe to be saved
        folder_name(str): The name of the folder
        file_name(str): The name of the file
    """
    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    df.to_parquet(folder_name + '/' + file_name + '.parquet', index=False)


def read_parquet(folder_name: str, file_name: str) -> pd.DataFrame:
    """Given a folder name and file name, we will load the parquet file.

    Args:
        folder_name(str): the name of the folder
        file_name(str): name of the file

    Returns:
        pd.DataFrame: the loaded csv file
    """
    path = folder_name + '/' + file_name + '.parquet'

    if not os.path.isfile(path):
        raise FileNotFoundError('File not found: ' + path)

    else:
        df = pd.read_parquet(path)
        return df


def save_dict(dictionary: dict, folder_name: str, file_name: str) -> None:
    """Save a dictionary to a json file

    Args:
        dictionary (dict): The dictionary to be saved
        folder_name (str): The name of the folder
        file_name (str): The name of the file
    """

    os.makedirs(folder_name, exist_ok=True)

    path = folder_name + '/' + file_name + '.json'

    with open(path, 'w') as file:
        json.dump(dictionary, file)


def read_dict(folder_name: str, file_name: str) -> dict:
    """Read a dictionary from a json file

    Args:
        folder_name (str): The name of the folder
        file_name (str): The name of the file

    Returns:
        dict: The dictionary
    """

    path = folder_name + '/' + file_name + '.json'

    with open(path, 'r') as file:
        dictionary = json.load(file)

    return dictionary


def save_pickle(obj: object, folder_name: str, file_name: str) -> None:
    """Save a dictionary to a json file

    Args:
        obj (object): The object to be saved
        folder_name (str): The name of the folder
        file_name (str): The name of the file
    """

    os.makedirs(folder_name, exist_ok=True)

    path = folder_name + '/' + file_name + '.pkl'

    with open(path, 'wb') as file:
        pickle.dump(obj, file)


def load_pickle(folder_name: str, file_name: str) -> object:
    """Read a dictionary from a json file

    Args:
        folder_name (str): The name of the folder
        file_name (str): The name of the file

    Returns:
        object: The object
    """

    path = folder_name + '/' + file_name + '.pkl'

    with open(path, 'rb') as file:
        obj = pickle.load(file)

    return obj
