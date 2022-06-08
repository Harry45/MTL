"""
Description: This file is for calculating the evaluation metrics and other quantities.
"""

# Author: Arrykrishna Mootoovaloo
# Date: April 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Project: Multi-Task Learning for Galaxy Zoo

import os
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, precision_score, recall_score

# our scripts and functions
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


def metrics_multilabel(predictions: str, ground_truth: str, save: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    # Hamming loss per class
    metrics_class = hamming_per_class(test, y_pred)

    if save:
        folder = 'results/metrics'
        os.makedirs(folder, exist_ok=True)
        hp.save_csv(metrics_class, folder, 'MTL_metrics_class')
        hp.save_csv(metrics_task, folder, 'MTL_metrics_task')

    return metrics_class, metrics_task
