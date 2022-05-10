"""
Description: This file is for processing the data to an appropriate format.
"""

# Author: Arrykrishna Mootoovaloo
# Date: April 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Project: Multi-Task Learning for Galaxy Zoo

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# our own functions and scripts
import utils.helpers as hp
import settings as st


def find_labels(tasks: dict) -> np.ndarray:
    """Find the labels of a galaxy given the outputs from the neural network.

    The tasks is a dictionary in the following format:

    tasks = {
        'task_1' : [1, 0, 0]
        'task_2' : [0, 1, 0]
        .
        .
        .
        'task_10': [0, 0, 1, 0, 0]
    }

    Args:
        tasks (dict): A dictionary with outputs from the neural network.

    Returns:
        np.ndarray: A numpy array consisting of the labels.
    """

    labels = {k: np.asarray(v) for k, v in st.LABELS.items()}
    tasks = {k: np.asarray(v) for k, v in tasks.items()}

    record_labels = pd.DataFrame(columns=['task_' + str(i + 1) for i in range(st.NUM_TASKS)])

    # Sometimes, there can be more than 1 label (due to equal probability by volunteers' votes)
    # If this happens, we pick the first selected label, hence [0] below.
    record_labels.at[0, 'task_2'] = list(labels['task_1'][tasks['task_1'] == 1])

    if tasks['task_1'][0] == 1:

        record_labels.at[0, 'task_2'] = list(labels['task_2'][tasks['task_2'] == 1])
        record_labels.at[0, 'task_2'] = list(labels['task_4'][tasks['task_4'] == 1])

    elif tasks['task_1'][1] == 1:
        record_labels.at[0, 'task_2'] = list(labels['task_3'][tasks['task_3'] == 1])

        if tasks['task_3'][0] == 1:
            record_labels.at[0, 'task_2'] = list(labels['task_5'][tasks['task_5'] == 1])
            record_labels.at[0, 'task_2'] = list(labels['task_4'][tasks['task_4'] == 1])

        else:
            record_labels.at[0, 'task_2'] = list(labels['task_6'][tasks['task_6'] == 1])
            record_labels.at[0, 'task_2'] = list(labels['task_7'][tasks['task_7'] == 1])

            if tasks['task_7'][0] == 1:
                record_labels.at[0, 'task_2'] = list(labels['task_8'][tasks['task_8'] == 1])
                record_labels.at[0, 'task_2'] = list(labels['task_9'][tasks['task_9'] == 1])
                record_labels.at[0, 'task_2'] = list(labels['task_10'][tasks['task_10'] == 1])
                record_labels.at[0, 'task_2'] = list(labels['task_4'][tasks['task_4'] == 1])

            else:
                record_labels.at[0, 'task_2'] = list(labels['task_10'][tasks['task_10'] == 1])
                record_labels.at[0, 'task_2'] = list(labels['task_4'][tasks['task_4'] == 1])

    return record_labels


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

    # the labels
    labels = vote_fraction.copy()

    # rename the columns according to labels in Decision Tree
    labels.rename(st.MAPPING, axis=1, inplace=True)

    # Order the columns according the tasks defined
    labels = labels[st.TASKS_ORDERED]

    rec = []
    for i in range(st.NUM_TASKS):

        test = labels[st.LABELS['task_' + str(i + 1)]]

        out = test.eq(test.max(axis=1), axis=0) * 1

        rec.append(out)

    # get the labels
    labels = pd.concat(rec, axis=1)

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
