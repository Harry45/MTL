"""
Description: This file is for processing the data to an appropriate format.
"""

# Author: Arrykrishna Mootoovaloo
# Date: April 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Project: Multi-Task Learning for Galaxy Zoo

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# our own functions and scripts
import utils.helpers as hp
import settings as st


def generate_labels(dataframe: pd.DataFrame, save: bool = False) -> pd.DataFrame:
    """Process the vote fraction and turn them into labels.

    Args:
        dataframe (pd.DataFrame): Name of the file which we want to process
        save (bool, optional): Option to save the file. Defaults to False.

    Raises:
        FileNotFoundError: Raises an error if file is not found.

    Returns:
        pd.DataFrame: A pandas dataframe consisting of the labels.
    """
    # number of columns in the dataframe
    ncols = dataframe.shape[1]

    # the vote fraction
    vote_fraction = dataframe[dataframe.columns[['fraction' in dataframe.columns[i] for i in range(ncols)]]]

    # generate the labels
    labels = vote_fraction.copy()
    labels[vote_fraction >= 0.5] = 1
    labels[vote_fraction < 0.5] = 0

    # we fill the NaN with -100 (we will be using cross-entropy later where we
    # can specify ignore_index = -100)
    labels.fillna(-100, inplace=True)

    # convert to integer
    labels = labels.astype(int)

    # keep the image names and locations in the file which contains the labels
    labels = pd.concat([dataframe[['iauname', 'png_loc']], labels], axis=1)

    if save:
        path = os.path.join(st.DATA_DIR, 'descriptions')
        hp.save_parquet(labels, path, 'labels')

    return labels


def split_data(dataframe: pd.DataFrame, val_size: float = 0.20, test_size: float = 0.20, save: bool = False) -> dict:
    """Split the data into training and validation size for assessing the performance of the network.

    Args:
        dataframe (pd.DataFrame): DataFrame consisting of the labels
        val_size (float, optional): The size of the validation set, a number between 0 and 1. Defaults to 0.20.
        test_size (float, optional): The size of the testing set, a number between 0 and 1. Defaults to 0.20.
        save (bool): Choose if we want to save the outputs generated. Defaults to False.

    Returns:
        dict: A dictionary consisting of the training and validation data.
    """

    # compute the training size
    train_size = 1.0 - val_size - test_size

    assert train_size > 0, "The validation and/or test size is too large."

    # split the data into train and test
    dummy, test = train_test_split(dataframe, test_size=test_size)

    # the validation size needs to be updated based on the first split
    val_new = val_size * dataframe.shape[0] / dummy.shape[0]

    # then we generate the training and validation set
    train, validate = train_test_split(dummy, test_size=val_new)

    # reset the index (not required, but just in case)
    test.reset_index(drop=True, inplace=True)
    train.reset_index(drop=True, inplace=True)
    validate.reset_index(drop=True, inplace=True)

    # store the dataframes in the dictionary
    record = {'train': train, 'validate': validate, 'test': test}

    if save:
        path = os.path.join(st.DATA_DIR, 'ml')
        hp.save_pd_csv(record['train'], path, 'train')
        hp.save_pd_csv(record['validate'], path, 'validate')
        hp.save_pd_csv(record['test'], path, 'test')

    return record


def correct_location(csv: str, save: bool = False, **kwargs) -> pd.DataFrame:
    """Rename the columns containing the image location to the right one.

    Args:
        csv (str): the name of the csv file
        save (bool): save the file if we want to. Defaults to False.

    Returns:
        pd.DataFrame: a dataframe consisting of the corrected location.
    """
    dataframe = hp.load_csv(st.ZENODO, csv)

    # the number of objects
    nobjects = dataframe.shape[0]

    # The png images are in the folder png/Jxxx/ rather than dr5/Jxxx/
    locations = [dataframe.png_loc.values[i][4:] for i in range(nobjects)]
    dataframe.png_loc = locations

    # check if all files exist
    imgs_exists = [int(os.path.isfile(st.DECALS + '/' + locations[i])) for i in range(nobjects)]
    imgs_exists = pd.DataFrame(imgs_exists, columns=['exists'])
    dataframe = pd.concat([dataframe, imgs_exists], axis=1)

    if save:
        filename = kwargs.pop('filename')
        folder = os.path.join(st.DATA_DIR, 'descriptions')
        hp.save_parquet(dataframe, folder, filename)

    return dataframe
