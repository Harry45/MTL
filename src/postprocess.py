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
import src.processing as sp


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

    # the different column names
    tags = list(test.columns[2:])

    # the binary predictions for each tag
    y_pred = np.array(pred.values, dtype=int)
    y_true = np.array(test.values[:, 2:], dtype=int)

    # create an empty dataframe to store the metrics
    metrics = pd.DataFrame(columns=['hamming_loss'], index=tags)

    hl_list = list()

    for i in range(len(tags)):
        hl_list.append(hamming_loss(y_true[:, i], y_pred[:, i]))

    metrics['hamming_loss'] = hl_list

    # overall metrics
    overall = pd.DataFrame(columns=['accuracy', 'hamming', 'precision', 'recall', 'f1'], index=['overall'])
    overall['accuracy'] = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    overall['hamming'] = hamming_loss(y_true, y_pred)
    overall['precision'] = precision_score(y_true=y_true, y_pred=y_pred, average='samples')
    overall['recall'] = recall_score(y_true=y_true, y_pred=y_pred, average='samples')
    overall['f1'] = f1_score(y_true=y_true, y_pred=y_pred, average='samples')

    if save:
        folder = 'results/metrics'
        os.makedirs(folder, exist_ok=True)
        hp.save_csv(metrics, folder, 'ML_metrics')
        hp.save_csv(overall, folder, 'ML_overall')

    return metrics, overall
