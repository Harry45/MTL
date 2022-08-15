"""
Description: This file is for calculating the evaluation metrics and other quantities.
"""

# Author: Arrykrishna Mootoovaloo
# Date: April 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Project: Multi-Task Learning for Galaxy Zoo

import os
import torch
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, precision_score, recall_score

# our scripts and functions
import src.processing as sp
import utils.helpers as hp
import settings as st


def labels_to_task(dataframe: pd.DataFrame, index: int = 0) -> dict:
    """Converts the labels to the task, that is, the task that the label belongs to.

    Args:
        dataframe (pd.DataFrame): A dataframe of the labels, with the first two
        columns corresponding to name and png locations.
        index (int, optional): The row we want to process. Defaults to 0.

    Returns:
        dict : A dictionary of the task that the label belongs to.
    """

    # the first two column names are the name and png locations
    labels = dataframe.iloc[index, 2:]

    label_dict = dict()
    for i in range(st.NUM_TASKS):
        task = labels[st.LABELS['task_' + str(i + 1)]].values.astype(int)
        label_dict['task_' + str(i + 1)] = task

    return label_dict


def hamming_per_class(test: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Calculates the hamming loss for each class.

    Args:
        test (pd.DataFrame): The ground truth.
        pred (np.ndarray): The predictions.

    Returns:
        pd.DataFrame: A dataframe containing the hamming loss for each class.
    """

    # the different column names
    tags = list(test.columns[2:])

    # the ground truths
    y_true = np.array(test.values[:, 2:], dtype=int)

    # create an empty dataframe to store the metrics
    metrics_class = pd.DataFrame(columns=['hamming_loss'], index=tags)

    hl_list = list()
    for i, j in enumerate(tags):
        ham_loss = hamming_loss(y_true[:, i], pred[:, i])
        hl_list.append(ham_loss)
        print(f'{j:30s}: {ham_loss:.3f}')

    metrics_class['hamming_loss'] = hl_list

    return metrics_class


def hamming_per_task(test: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Calculates the hamming loss for each task.

    Args:
        test (pd.DataFrame): The ground truth.
        pred (pd.DataFrame): The predictions.

    Returns:
        pd.DataFrame: A dataframe containing the hamming loss for each task.
    """

    # the labels for the tasks
    task_labels = ['task_' + str(i + 1) for i in range(st.NUM_TASKS)]

    # create an empty dataframe to store the metrics
    metrics_task = pd.DataFrame(columns=['hamming_loss'], index=task_labels)

    # create an empty list to store the hamming loss for each task
    task_list = list()

    for i in range(st.NUM_TASKS):
        pred_task = np.vstack(pred['task_' + str(i + 1)].values)
        test_task = test[st.LABELS['task_' + str(i + 1)]].values
        ham_loss = hamming_loss(test_task, pred_task)
        task_list.append(ham_loss)
        print(f'Task {i+1:2d}: {ham_loss:.3f}')

    metrics_task['hamming_loss'] = task_list

    return metrics_task


def metric_multilabel(predictions: str, ground_truth: str, save: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculates the metrics for the multilabel problem.

    Args:
        predictions (str): The name of the file containing the predictions.
        ground_truth (str): The name of the file containing the ground truth.
        save (bool, optional): Whether to save the results. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of dataframes containing the metrics.
    """

    pred = hp.load_csv('results', predictions)
    test = hp.load_csv('ml', ground_truth)

    assert pred.shape[0] == test.shape[0], "The shapes of the predictions and ground truth do not match."

    # the binary predictions for each tag
    y_pred = np.array(pred.values, dtype=int)
    y_true = np.array(test.values[:, 2:], dtype=int)

    # create an empty dataframe to store the metrics
    metrics = hamming_per_class(test, y_pred)

    # overall metrics
    overall = pd.DataFrame(columns=['accuracy', 'hamming', 'precision', 'recall', 'f1'], index=['overall'])
    overall['accuracy'] = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    ham_loss = hamming_loss(y_true, y_pred)
    overall['hamming'] = ham_loss
    overall['precision'] = precision_score(y_true=y_true, y_pred=y_pred, average='samples')
    overall['recall'] = recall_score(y_true=y_true, y_pred=y_pred, average='samples')
    overall['f1'] = f1_score(y_true=y_true, y_pred=y_pred, average='samples')

    print(f'Overall Hamming loss is: {ham_loss:.3f}')

    if save:
        folder = 'results/metrics'
        os.makedirs(folder, exist_ok=True)
        hp.save_csv(metrics, folder, 'ML_metrics')
        hp.save_csv(overall, folder, 'ML_overall')

    return metrics, overall


def metric_multitask(predictions: str, ground_truth: str, save: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculates the metrics for the multitask problem.

    Args:
        predictions (str): The name of the file containing the predictions.
        ground_truth (str): The name of the file containing the ground truth.
        save (bool, optional): Whether to save the results. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of dataframes containing the metrics.
    """

    # the predicted labels
    pred = hp.load_pickle('results', predictions)

    # the dataframe containing the ground truth (the first two columns are the name and png locations)
    test = hp.load_csv('ml', ground_truth)

    # the predictions in a pandas dataframe
    pred_df = pd.DataFrame(pred)

    assert pred_df.shape[0] == test.shape[0], "The shapes of the predictions and ground truth do not match."

    # overall Hamming Loss
    # --------------------
    tasks = [np.vstack(pred_df['task_' + str(i + 1)].values) for i in range(st.NUM_TASKS)]
    y_pred = np.concatenate(tasks, axis=1)

    print(f'Overall Hamming loss is: {hamming_loss(test.iloc[:,2:].values, y_pred):.3f}')
    print('-' * 50)

    # Hamming loss per task
    metrics_task = hamming_per_task(test, pred_df)
    print('-' * 50)

    # Hamming loss per class
    metrics_class = hamming_per_class(test, y_pred)

    if save:
        folder = 'results/metrics'
        os.makedirs(folder, exist_ok=True)
        hp.save_csv(metrics_class, folder, 'MTL_metrics_class')
        hp.save_csv(metrics_task, folder, 'MTL_metrics_task')

    return metrics_class, metrics_task


def compare_labels_ml(
        dataloader: torch.utils.data.DataLoader, idx: int, pred: pd.DataFrame, savefig: bool = False) -> Tuple[
        np.ndarray, np.ndarray]:
    """Compare the predicted labels with the ground truth for the multi-label
    learning case.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader containing the data.
        idx (int): The index of the image to be compared.
        pred (pd.DataFrame): The predictions.
        savefig (bool, optional): Option to save the figure. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The predicted and ground truth labels.
    """

    # the labels
    columns = np.array(dataloader.dataset.desc.columns[2:])

    # the filename
    filename = dataloader.dataset.desc.iauname.iloc[idx]

    # the data
    data = dataloader.dataset[idx]

    # the predicted labels
    labels_pred = columns[pred.iloc[idx].values == 1]

    # the ground truth
    labels_test = columns[data[1] == 1]

    name = 'copper'
    plt.figure(figsize=(4, 4))
    plt.imshow(data[0].permute(1, 2, 0), cmap=plt.get_cmap(name))
    plt.axis('off')

    if savefig:
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/{filename}.pdf', bbox_inches='tight')
    plt.show()

    print('Predictions')
    print('-' * 11)
    for i, element in enumerate(labels_pred):
        print(f'Tag {i:2d}: {element}')

    print()
    print('Volunteers')
    print('-' * 10)
    for i, element in enumerate(labels_test):
        print(f'Tag {i:2d}: {element}')

    return labels_test, labels_pred


def labels_per_task(dataframe: pd.DataFrame, index: int = 0) -> dict:
    """Creates a dictionary where the keys are the tasks and the values are the
    numeric values (0 or 1).

    Args:
        dataframe (pd.DataFrame): a dataframe with the labels.
        index (int, optional): the row we want to choose from the dataframe. Defaults to 0.

    Returns:
        dict: a dictionary where each task has a list of labels, for example,
        {'task_1': [0, 1, 0]}
    """

    # the first two column names are the name and png locations
    labels = dataframe.iloc[index, 2:]

    label_dict = dict()
    for i in range(st.NUM_TASKS):
        task = labels[st.LABELS['task_' + str(i + 1)]].values.astype(int)
        label_dict['task_' + str(i + 1)] = task

    return label_dict


def build_tree(test_set: pd.DataFrame, pred_set: pd.DataFrame,
               save: bool, fname: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Builds a tree for the multitask learning case.

    Args:
        test_set (pd.DataFrame): The ground truth.
        pred_set (pd.DataFrame): The predictions.
        save (bool): Whether to save the tree.
        fname (str): The name of the file to save the tree.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The ground truth and predictions.
    """

    # number of test point
    ntest = test_set.shape[0]

    # create empty lists to store the trees
    tree_pred = list()
    tree_test = list()

    for idx in range(ntest):

        # re-write the test point in a dictionary (label per task)
        test_point = labels_per_task(test_set, idx)

        # find the labels (ground truths and test point)
        pred_labels = sp.find_labels(pred_set[idx])
        test_labels = sp.find_labels(test_point)

        # record the trees
        tree_pred.append(pred_labels)
        tree_test.append(test_labels)

    # generate the dataframe with the trees
    tree_pred_df = pd.concat(tree_pred)
    tree_test_df = pd.concat(tree_test)

    # reset the indices
    tree_pred_df.reset_index(drop=True, inplace=True)
    tree_test_df.reset_index(drop=True, inplace=True)

    # store the trees
    if save:
        hp.save_pd_csv(tree_pred_df, 'results', f'tree_pred-{fname}')
        hp.save_pd_csv(tree_test_df, 'results', f'tree_test-{fname}')

    return tree_pred_df, tree_test_df
