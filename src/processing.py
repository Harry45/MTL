# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: This file is for processing the data (tags) and making selections.
# Project: One/Few-Shot Learning for Galaxy Zoo

import os
import shutil
import pandas as pd
import numpy as np
import sklearn.model_selection as sm

# our own functions and scripts
import utils.helpers as hp
import settings as st


def generate_labels(fname: str, save: bool = False) -> pd.DataFrame:
    """Process the vote fraction and turn them into labels.

    Args:
        fname (str): Name of the file which we want to process
        save (bool, optional): Option to save the file. Defaults to False.

    Raises:
        FileNotFoundError: Raises an error if file is not found.

    Returns:
        pd.DataFrame: A pandas dataframe consisting of the labels.
    """

    desc = hp.read_parquet(st.data_dir, 'descriptions/' + fname)

    # number of columns in the dataframe
    ncols = desc.shape[1]

    # the vote fraction
    vote_fraction = desc[desc.columns[['fraction' in desc.columns[i] for i in range(ncols)]]]

    # generate the labels
    labels = vote_fraction.copy()
    labels[vote_fraction >= 0.5] = 1
    labels[vote_fraction < 0.5] = 0

    # we fill the NaN with -100 (we will be using cross-entropy later where we
    # can specify ignore_index = -100)
    labels.fillna(-100, inplace=True)

    # keep the image names and locations in the file which contains the labels
    labels = pd.concat([desc[['iauname', 'png_loc']], labels], axis=1)

    if save:
        hp.save_parquet(labels, st.data_dir + '/descriptions', 'labels')

    return labels


def correct_location(csv: str, save: bool = False, **kwargs) -> pd.DataFrame:
    """Rename the columns containing the image location to the right one.

    Args:
        csv (str): the name of the csv file
        save (bool): save the file if we want to. Defaults to False.

    Returns:
        pd.DataFrame: a dataframe consisting of the corrected location.
    """
    dataframe = hp.load_csv(st.zenodo, csv)

    # the number of objects
    nobjects = dataframe.shape[0]

    # The png images are in the folder png/Jxxx/ rather than dr5/Jxxx/
    locations = [dataframe.png_loc.values[i][4:] for i in range(nobjects)]
    dataframe.png_loc = locations

    # check if all files exist
    imgs_exists = [int(os.path.isfile(st.decals + '/' + locations[i])) for i in range(nobjects)]
    imgs_exists = pd.DataFrame(imgs_exists, columns=['exists'])
    dataframe = pd.concat([dataframe, imgs_exists], axis=1)

    if save:
        filename = kwargs.pop('filename')
        hp.save_parquet(dataframe, st.data_dir + '/descriptions', filename)

    return dataframe


def filtering(dataframe: pd.DataFrame, dictionary: dict, save: bool = False, **kwargs) -> pd.DataFrame:
    """Given a dictionary of filters, filter the dataframe. For example,

    dictionary = {'has-spiral-arms_yes_fraction' : 0.75, 'has-spiral-arms_yes' : 20}

    means we have at least 20 volunteers, who have voted for spiral arms, and the
    fraction of those who voted for spiral arms is at least 0.75.

    Note that the keys in the dictionary are the column names in the dataframe.

    Args:
        df (pd.DataFrame): A pandas dataframe with the metadata
        dictionary (dict): A dictionary of filters.
        save (bool): Option to save the outputs. Defaults to False.

    Returns:
        pd.DataFrame: A pandas dataframe with the filtered data.
    """

    # number of objects in the dataframe
    nobjects = dataframe.shape[0]

    # items in the dictionary
    items = list(dictionary.items())

    condition = [True] * nobjects

    for item in items:
        condition &= dataframe[item[0]] > item[1]

    # apply condition and reset index
    df_sub = dataframe[condition]
    df_sub.reset_index(inplace=True, drop=True)

    if save:
        filename = kwargs.pop('filename')
        hp.save_parquet(df_sub, st.data_dir + '/descriptions', filename)

    return df_sub


def subset_df(dataframe: pd.DataFrame, nsubjects: int, random: bool = False,
              save: bool = False, **kwargs) -> pd.DataFrame:
    """Generate a subset of objects, for example, 2 000 out of 10 000 spirals.

    Args:
        dataframe (pd.DataFrame): A dataframe consisting of specific objects, for example, spirals.
        nsubjects (int): The number of subjects we want to pick.
        random (bool): We can set this to True, if we want to pick the subjects randomly.
        save (bool): Option to save the outputs. Defaults to False.

    Returns:
        pd.DataFrame: A pandas dataframe consisting of a subset of images.
    """

    # total number of objects
    total = dataframe.shape[0]

    assert nsubjects <= total, 'The number of subjects requested is larger than the available number of objects.'

    if random:
        idx = np.random.choice(total, nsubjects, replace=False)

    else:
        idx = range(nsubjects)

    df_sub = dataframe.iloc[idx]
    df_sub.reset_index(inplace=True, drop=True)

    if save:
        filename = kwargs.pop('filename')
        hp.save_parquet(df_sub, st.data_dir + '/descriptions', filename)

    return df_sub


def copy_images(dataframe: pd.DataFrame, foldername: str) -> None:
    """Copy images from Mike's folder to our working directory.

    Args:
        df (pd.DataFrame): A dataframe consisting of specific objects, for example, spiral
        foldername (str): Name of the folder where we want to copy the images
    """

    # number of objects
    nobjects = dataframe.shape[0]

    # create a folder where we want to store the images
    folder = st.data_dir + '/' + 'images' + '/' + foldername + '/'

    # create the different folders if they do not exist (remove them if they exist already)
    if os.path.exists(folder):

        # delete the folder first if it exists
        shutil.rmtree(folder)

    # then create a new one
    os.makedirs(folder)

    counts = 0
    # fetch the data from Mike's directory
    for i in range(nobjects):

        decals_file = st.decals + '/' + dataframe['png_loc'].iloc[i]

        if os.path.isfile(decals_file):
            cmd = f'cp {decals_file} {folder}'
            os.system(cmd)
            counts += 1

    print(f'{counts} images saved to {folder}')


def split_data(tag_names: list, val_size: float = 0.20, test_size: float = 0.20, save: bool = False) -> dict:
    """Split the data into training and validation size for assessing the performance of the network.

    Args:
        tag_names (list): A list of the tag names, for example, elliptical, ring, spiral
        val_size (float, optional): The size of the validation set, a number between 0 and 1. Defaults to 0.20.
        test_size (float, optional): The size of the testing set, a number between 0 and 1. Defaults to 0.20.
        save (bool): Choose if we want to save the outputs generated. Defaults to False.

    Returns:
        dict: A dictionary consisting of the training and validation data.
    """

    # compute the training size
    train_size = 1.0 - val_size - test_size

    assert train_size > 0, "The validation and/or test size is too large."

    record = {}

    for item in tag_names:

        # load the csv file
        tag_file = hp.read_parquet(st.data_dir, 'descriptions/subset_' + item)

        # split the data into train and test
        dummy, test = sm.train_test_split(tag_file, test_size=test_size)

        # the validation size needs to be updated based on the first split
        val_new = val_size * tag_file.shape[0] / dummy.shape[0]

        # then we generate the training and validation set
        train, validate = sm.train_test_split(dummy, test_size=val_new)

        # reset the index (not required, but just in case)
        test.reset_index(drop=True, inplace=True)
        train.reset_index(drop=True, inplace=True)
        validate.reset_index(drop=True, inplace=True)

        # store the dataframes in the dictionary
        record[item] = {'train': train, 'validate': validate, 'test': test}

        if save:
            hp.save_pd_csv(record[item]['train'], st.data_dir + '/' + 'ml/train', item)
            hp.save_pd_csv(record[item]['validate'], st.data_dir + '/' + 'ml/validate', item)
            hp.save_pd_csv(record[item]['test'], st.data_dir + '/' + 'ml/test', item)

    return record


def move_data(subset: str, object_type: str) -> None:
    """Move data to the right folder.

    Args:
        subset (str) : validate or train or test
        object_type (str): name of the object, for example, 'spiral', we want to move
    """

    assert subset in ['validate', 'test', 'train'], "Typical group in ML: validate, train, test"

    # the Machine Learning set (validate, train, test)
    ml_set = hp.load_csv(st.data_dir + '/ml/' + subset, object_type)

    # number of objects we have
    nobject = ml_set.shape[0]

    # folder where we want to store the images
    folder = st.data_dir + '/' + 'ml' + '/' + subset + '_images' + '/' + object_type + '/'

    # create the different folders if they do not exist (remove them if they exist already)
    if os.path.exists(folder):

        # delete the folder first if it exists
        shutil.rmtree(folder)

    # then create a new one
    os.makedirs(folder)

    # copy the data from images/item to categories/train/item
    for j in range(nobject):

        file = st.data_dir + '/' + 'images' + '/' + object_type + '/' + ml_set.iauname.iloc[j] + '.png'

        if os.path.isfile(file):
            cmd = f'cp {file} {folder}'
            os.system(cmd)


def images_train_validate_test(tag_names: list) -> None:
    """Read the csv file for a particular tag and copy the images in their respective folders

    Args:
        tag_names (list): A list of the tag names, for example, elliptical, ring, spiral
    """

    for item in tag_names:
        for subset in ['validate', 'train', 'test']:
            move_data(subset, item)


def copy_test_images(images: list, target_folder: str):
    """Copy a single image to the target folder

    Args:
        images (list): a list with the names of the object (png_loc, that is, Jxxx/image)
        target_folder (str): the folder where we want to test the image
    """

    # we also want the details of the galaxy
    dr5_desc = hp.read_parquet(st.data_dir, 'descriptions/decals_5_votes')

    for image in images:
        desc = dr5_desc[dr5_desc['png_loc'] == image]

        # save the description of the galaxy
        obj = os.path.split(image)[-1][:-4]
        hp.save_pickle(desc, 'test-images', obj)

        # copy image
        full_path = os.path.join(st.decals, image)

        if not os.path.exists:
            raise FileNotFoundError(f'{full_path} does not exist')

        cmd = f'cp {full_path} {target_folder}'
        os.system(cmd)
