"""
Description: Prediction for the multi-label case.
"""

# Author: Arrykrishna Mootoovaloo
# Date: April 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Project: Multi-Task Learning for Galaxy Zoo
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

# our scripts and functions
from src.network import MultiLabelNet
from src.dataset import DECaLSDataset
import settings as st
import utils.helpers as hp


def predict_class(output: torch.Tensor) -> torch.Tensor:
    """Convert the logits into specific class. Number below 0 is assigned 0 else
    assigned 1.

    Args:
        output (torch.Tensor): the output from the neural network

    Returns:
        torch.Tensor: the outputs in binary format
    """
    # modify the output from the neural network
    output = output.cpu().detach().numpy().reshape(-1)
    output[output >= 0] = 1
    output[output < 0] = 0
    output = output.astype(int)
    return output


def predict_probability(output: torch.Tensor) -> torch.Tensor:
    """Predict the probability of a specific class. Convert logits to
    probabilities.

    Args:
        output (torch.Tensor): the output from the neural network

    Returns:
        torch.Tensor: the outputs as probabilities
    """
    output = output.clone()
    probability = torch.sigmoid(output)
    probability = probability.cpu().detach().numpy().reshape(-1)
    return probability


# we will normally evaluate on CPU (to maend if we want to predict on GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the model
model_path = os.path.join('/data/phys-zooniverse/phys2286', 'Models', 'ml-models-2022-5-25')
loaded_model = torch.load(model_path + '/' + 'resnet_18_multilabel_29.pth')
model.to(device)
model.load_state_dict(loaded_model)
model.eval()

test_dataset = DECaLSDataset(mode='test', augment=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

record_outputs = list()
record_prob = list()

count = 0
ndata = len(test_dataset)

for images, targets in test_loader:

    # images and targets to device (CPU or GPU)
    images, targets = map(lambda x: x.to(device), [images, targets])

    # compute the outputs
    outputs = model(images)

    # convert the logits into binary
    out = predict_class(outputs)

    # convert the logits into probabilities
    prob = predict_probability(outputs)

    # record the results
    record_outputs.append(out)
    record_prob.append(prob)

    # augment count
    count += 1

    if (count + 1) % 5000 == 0:
        print("Processed {}/{}".format(count + 1, ndata))

# convert the results to dataframes
class_df = pd.DataFrame(record_outputs, columns=['f' + str(i + 1) for i in range(st.NCLASS)])
prob_df = pd.DataFrame(record_prob, columns=['f' + str(i + 1) for i in range(st.NCLASS)])

# store the outputs
hp.save_pd_csv(class_df, 'results', 'ML_predictions_class')
hp.save_pd_csv(prob_df, 'results', 'ML_predictions_prob')
