"""
Description: Train the network for the multi-label case.
"""

# Author: Arrykrishna Mootoovaloo
# Date: April 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Project: Multi-Task Learning for Galaxy Zoo

import os
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# our scripts and functions
from src.network import MultiLabelNet
from src.dataset import DECaLSDataset
import settings as st

date = datetime.datetime.now()
today = str(date.year) + '-' + str(date.month) + '-' + str(date.day)

out_path = './output/'
model_path = '/data/phys-zooniverse/phys2286/Models/ml-models-' + today + '/'

# make the folders
os.makedirs(out_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# Set device to CUDA if a CUDA device is available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create the dataloader
train_dataset = DECaLSDataset(mode='train', augment=False, multi_task=False)
val_dataset = DECaLSDataset(mode='validate', augment=False, multi_task=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

# define the model
model = MultiLabelNet(backbone="resnet18")
model = nn.DataParallel(model)
model.to(device)

# set the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1E-4, weight_decay=1E-5)

# to assign weights to this loss function
weights = torch.tensor(st.CLASS_WEIGHTS).to(device)

# loss function
# criterion = nn.MultiLabelSoftMarginLoss(weight=weights, reduction='mean')
# criterion = nn.BCEWithLogitsLoss(weight=weights, reduction='mean')
criterion = nn.BCEWithLogitsLoss(reduction='mean')

writer = SummaryWriter(os.path.join(out_path, "summary"))

epochs = 30

for epoch in range(epochs):
    print("Epoch [{} / {}]".format(epoch + 1, epochs))
    model.train()

    losses = []

    # Training Loop Start
    for images, targets in train_loader:
        images, targets = map(lambda x: x.to(device), [images, targets])

        outputs = model(images)
        loss = criterion(outputs, targets.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    train_loss = sum(losses) / len(losses)
    writer.add_scalar('train_loss', train_loss, epoch)

    print(f"Training   : Loss={train_loss:.2e}")
    # Training Loop End

    # Evaluation Loop Start
    model.eval()

    losses = []

    for images, targets in val_loader:
        images, targets = map(lambda x: x.to(device), [images, targets])

        outputs = model(images)
        loss = criterion(outputs, targets.float())

        losses.append(loss.item())

    val_loss = sum(losses) / len(losses)
    writer.add_scalar('val_loss', val_loss, epoch)

    print(f"Validation : Loss={val_loss:.2e}")
    print("-" * 30)

    torch.save(model.state_dict(), model_path + 'resnet_18_multilabel_' + str(epoch) + '.pth')
