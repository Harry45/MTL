"""
Description: Prediction for the multi-task case.
"""

# Author: Arrykrishna Mootoovaloo
# Date: April 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Project: Multi-Task Learning for Galaxy Zoo

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# our scripts and functions
from src.network import MultiTaskNet
from src.dataset import DECaLSDataset
import settings as st
import utils.helpers as hp


def predict_labels(output: nn.ModuleDict) -> dict:
    """Predict the probability of a specific class. Convert logits to
    probabilities and assign 1 to the label within that specific task.

    Args:
        output (nn.ModuleDict): the output from the neural network

    Returns:
        dict: the predictions for each task
    """

    pred = {}
    for i in range(st.NUM_TASKS):

        # predictions for the i^th task (logits)
        logits = output['task_' + str(i + 1)]

        # convert to probability
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        probabilities = probabilities.view(-1)
        n_prob = probabilities.shape[0]

        # assign 1 to the label having maximum probability
        index = probabilities.max(0).indices.item()
        class_index = np.zeros(n_prob, dtype=int)
        class_index[index] = 1

        # record the predictions
        pred['task_' + str(i + 1)] = class_index

    return pred


# we will normally evaluate on CPU (to maend if we want to predict on GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the model
loaded_model = torch.load('../mtl-models/resnet_18_multitask_12.pth')
model = MultiTaskNet(backbone="resnet18", output_size=st.LABELS_PER_TASK, resnet_task=True)
model.to(device)
model.load_state_dict(loaded_model)
model.eval()

test_dataset = DECaLSDataset(mode='test', augment=False, multi_task=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

record_outputs = list()

count = 0
ndata = len(test_dataset)
step = int(0.1 * ndata)

for images, targets in test_loader:

    # images and targets to device (CPU or GPU)
    images = images.to(device)

    # compute the outputs
    outputs = model(images)

    # convert the logits into binary
    out = predict_labels(outputs)

    # record the results
    record_outputs.append(out)

    # augment count
    count += 1

    if (count + 1) % step == 0:
        print("Processed {}/{}".format(count + 1, ndata))

# store the outputs
hp.save_pickle(record_outputs, 'results', 'predictions_mtl')
