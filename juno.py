"""
Description: functions for the additional dataset (JUNO)
"""

# Author: Arrykrishna Mootoovaloo
# Date: September 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Project: Multi-Task Learning for Galaxy Zoo

import os
import shutil
import subprocess
import random
import datetime
from typing import Tuple
import matplotlib.pylab as plt
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms

# our scripts and functions
from src.fewshot import generate_labels_fewshot
from src.fewshot import shannon_entropy
import settings as st
import utils.helpers as hp

DATE = datetime.datetime.now()
TODAY = str(DATE.year) + '-' + str(DATE.month) + '-' + str(DATE.day)
TIME = str(DATE.hour) + ':' + str(DATE.minute)


def view_images(dataframe: pd.DataFrame, transformation: bool = False):
    """View the images in a dequential manner in order to annotate them.

    Args:
        dataframe (pd.DataFrame):a csv file with the list of images
        transformation (bool, optional): option to first transform the images
        before viewing them. Defaults to False.
    """

    if transformation:
        trans = st.TRANS
        transform = transforms.Compose(trans)

    for i, name in enumerate(dataframe['name'].values):
        filepath = 'juno/images-original/' + name
        image = Image.open(filepath).convert("RGB")

        plt.figure(figsize=(8, 8))
        if transformation:
            image = transform(image).float()
            plt.imshow(image[0])
        else:
            plt.imshow(image)
        plt.title(f'{i}: {name[:-4]}', fontdict={'fontsize': 20})
        plt.axis('off')
        plt.show()


def _copy_images_juno(labels: pd.DataFrame, nshot: int = 10, index: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Copy each category of objects in their respective folders.

    Args:
        labels (pd.DataFrame): the labelled data.
        nshot (int, optional): the number of shots. Defaults to 10.
        index (int, optional): the specific label. Defaults to 0.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The support and query sets.
    """
    subset = labels[labels['Targets'] == index]
    subset.reset_index(drop=True, inplace=True)

    assert subset.shape[0] > nshot, 'Number of shots more than available'
    nquery = subset.shape[0] - nshot
    print(f'Number of shots is {nshot}')
    print(f'Number of query is {nquery}')

    # create the support and query datsets
    randint = random.sample(range(0, subset.shape[0]), nshot)
    support = subset.iloc[randint]
    query = subset[~subset.index.isin(randint)]
    support.reset_index(drop=True, inplace=True)
    query.reset_index(drop=True, inplace=True)

    # create path for the support set
    path_support = f'juno/train/{str(nshot)}-shots/{st.FS_MAP[index]}/'
    if os.path.exists(path_support) and os.path.isdir(path_support):
        shutil.rmtree(path_support)
    os.makedirs(path_support, exist_ok=True)

    # copy images
    for i in range(nshot):
        path = 'juno/images-original/' + support['Objects'].values[i]
        subprocess.run(["cp", path, path_support], capture_output=True, text=True)

    for i in range(nquery):
        path = 'juno/images-original/' + query['Objects'].values[i]
        subprocess.run(["cp", path, st.PATH_QUERY], capture_output=True, text=True)

    return support, query


def copy_images_juno(labels: pd.DataFrame, nshot: int, save: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Copy the images to the correct folder

    Args:
        labels (pd.DataFrame): a csv file with the labels (integers)
        nshot (int): number of shots to use (recommend 10)
        save (bool): save the support and query dataframes

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: the support and query sets
    """

    # create the query path only once
    if os.path.exists(st.PATH_QUERY) and os.path.isdir(st.PATH_QUERY):
        shutil.rmtree(st.PATH_QUERY)
    os.makedirs(st.PATH_QUERY, exist_ok=True)

    # record the support and query dataframes
    support_rec = list()
    query_rec = list()

    # copy the images for each class
    for i in range(len(st.FS_CLASSES)):
        support, query = _copy_images_juno(labels, nshot, i)
        support_rec.append(support)
        query_rec.append(query)

    # record all the support and query objects
    support_df = pd.concat(support_rec)
    query_df = pd.concat(query_rec)

    support_df.reset_index(drop=True, inplace=True)
    query_df.reset_index(drop=True, inplace=True)

    # add the word labels
    support_df['Targets'] = support_df['Targets'].map(st.FS_MAP)
    query_df['Targets'] = query_df['Targets'].map(st.FS_MAP)

    support_df.columns = ['Objects', 'Labels']
    query_df.columns = ['Objects', 'Labels']

    if save:
        hp.save_pd_csv(query_df, 'juno', f'train/query_{str(nshot)}')
        hp.save_pd_csv(support_df, 'juno', f'train/support_{str(nshot)}')

    return support_df, query_df


def performance(folder: str, csvfile: str) -> float:
    """Calculates the accuracy with which the few shot learning performs

    Args:
        folder (str): name of the folder where the file is stored
        csvfile (str): name of the file (should contain True Labels and
        Predicted Labels)

    Returns:
        float: the accuracy
    """

    file = hp.load_csv(folder, csvfile)
    total = (file['True Labels'] == file['Predicted Labels']) * 1
    acc = sum(total) / len(total)

    return acc


def finetuning(model: nn.Module, loaders_training: dict, quant: dict, save: bool) -> nn.Module:
    """Fine tune the model and adapt it to the few shot learning task.

    Args:
        model (nn.Module): the pre-trained deep learning model.
        loaders_training (dict): a dictionary for the support and query data
        (training set).
        quant (dict): a dictionary with important quantities, for example,

        quant = {'lr': 1E-3,
        'weight_decay': 1E-5,
        'coefficient': 0.01,
        'nepochs': 150,
        'criterion': nn.CrossEntropyLoss()}

        where weight decay is a regularizer, coefficient is a regularization
        term for the Shannon Entropy calculation and so forth.

        save (bool): if True, the model will be save to the models directory

    Returns:
        nn.Module: the fine tuned model.
    """

    # choose an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=quant['lr'], weight_decay=quant['weight_decay'])

    losses_train = []

    # iterate over the whole dataset
    for epoch in range(quant['nepochs']):

        model.train()

        loss_support_rec = []
        loss_query_rec = []

        # calculate loss due to the support set
        for images, targets in loaders_training['support']:
            images, targets = map(lambda x: x.to(st.DEVICE), [images, targets])

            outputs = model(images)

            loss_support = quant['criterion'](outputs, targets.view(-1))

            optimizer.zero_grad()
            loss_support.backward()
            optimizer.step()

            loss_support_rec.append(loss_support.item())

        # calculate loss due to the query set (transductive learning)
        for images, targets in loaders_training['query']:
            images, targets = map(lambda x: x.to(st.DEVICE), [images, targets])

            outputs = model(images)

            loss_query = quant['coefficient'] * shannon_entropy(outputs)

            optimizer.zero_grad()
            loss_query.backward()
            optimizer.step()

            loss_query_rec.append(loss_query.item())

        # calculate the total loss (training set)
        total_loss = sum(loss_support_rec) / len(loss_support_rec) + sum(loss_query_rec) / len(loss_query_rec)
        losses_train.append(total_loss)
        print(f"Epoch [{epoch + 1} / {quant['nepochs']}]: Training Loss = {total_loss:.4f}")

    # save the values of the loss
    if save:
        nepochs = str(quant['nepochs'])
        hp.save_pickle(losses_train, 'results', f'loss_finetune_juno_train_{TODAY}_{nepochs}')
        torch.save(model.state_dict(), st.MODEL_PATH + f'finetune_juno_{TODAY}_{nepochs}.pth')

    return model


# if __name__ == '__main__':
    # labels = pd.read_csv('juno/labels-completed.csv', index_col=0)
    # support, query = copy_images_juno(labels, 10, True)
    # query, support = generate_labels_fewshot(nshot=10, save=True, train=True)
    # simple_acc = performance('juno', 'train/nearest_neighbour_10')
    # fine_acc = performance('juno', 'finetune_train_10_300')
    # print(f'Accuracy with simple shot method : {simple_acc:.2f}')
    # print(f'Accuracy with finetuning method  : {fine_acc:.2f}')
