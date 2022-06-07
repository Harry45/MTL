"""
Description: This file is for calculating the evaluation metrics and other quantities.
"""

# Author: Arrykrishna Mootoovaloo
# Date: April 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Project: Multi-Task Learning for Galaxy Zoo

import os
import pandas as pd

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
