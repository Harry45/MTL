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

# our scripts and functions
from src.network import MultiLabelNet
from src.dataset import DECaLSDataset
import settings as st

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

for images, targets in test_loader:
    images, targets = map(lambda x: x.to(device), [images, targets])

    outputs = model(images)
    loss = criterion(outputs, targets)

    print(targets)
    print(nn.Softmax(outputs))
    print('-'*100)
