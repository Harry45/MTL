"""
Description: Prediction for the multi-label case.
"""

# Author: Arrykrishna Mootoovaloo
# Date: April 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Project: Multi-Task Learning for Galaxy Zoo

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

# our scripts and functions
from src.network import MultiLabelNet
from src.dataset import DECaLSDataset
import settings as st
import utils.helpers as hp


def process_outputs(outputs, threshold=0.1):
    out = nn.functional.softmax(outputs, dim=1)
    out[out >= threshold] = 1
    out[out < threshold] = 0
    out = out.type(torch.int)
    out = out.cpu().detach().numpy().reshape(-1)
    return out


# we will normally evaluate on CPU (to maend if we want to predict on GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the model
loaded_model = torch.load('../ml-models/resnet_18_multilabel.pth')
model = MultiLabelNet(backbone="resnet18")
model.to(device)
model.load_state_dict(loaded_model)
model.eval()

test_dataset = DECaLSDataset(mode='test', augment=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# to assign weights to this loss function
weights = torch.tensor(st.CLASS_WEIGHTS).to(device)
criterion = nn.MultiLabelSoftMarginLoss(weight=weights, reduction='mean')

# criterion = nn.BCEWithLogitsLoss()

record_outputs = list()

for images, targets in test_loader:
    images, targets = map(lambda x: x.to(device), [images, targets])

    outputs = model(images)
    loss = criterion(outputs, targets)
    out = process_outputs(outputs)
    record_outputs.append(out)

record_df = pd.DataFrame(record_outputs, columns=['f'+str(i+1) for i in range(st.NCLASS)])

hp.save_pd_csv(record_df, 'results', 'predictions')
