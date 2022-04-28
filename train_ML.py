"""
Description: Train the network for the multi-label case.
"""

# Author: Arrykrishna Mootoovaloo
# Date: April 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Project: Multi-Task Learning for Galaxy Zoo

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# our scripts and functions
from src.network import MultiLabelNet
from src.dataset import DECaLSDataset
import settings as st

out_path = './output/'
os.makedirs(out_path, exist_ok=True)

# Set device to CUDA if a CUDA device is available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create the dataloader
train_dataset = DECaLSDataset(mode='train', augment=False)
val_dataset = DECaLSDataset(mode='validate', augment=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=False)

# define the model
model = MultiLabelNet(backbone="resnet18")
model.to(device)

# to assign weights to this loss function
weights = torch.tensor(st.CLASS_WEIGHTS)

# set the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1E-4, weight_decay=1E-5)
criterion = nn.MultiLabelSoftMarginLoss(weight=weights, reduction='mean')

writer = SummaryWriter(os.path.join(out_path, "summary"))

epochs = 2

for epoch in range(epochs):
    print("[{} / {}]".format(epoch + 1, epochs))
    model.train()

    losses = []

    # Training Loop Start
    for images, targets in train_loader:
        images, targets = map(lambda x: x.to(device), [imgages, targets])

        outputs = model(images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    train_loss = sum(losses) / len(losses)
    writer.add_scalar('train_loss', train_loss, epoch)

    print(f"Training: Loss={train_loss:.2f}")
    # Training Loop End

    # Evaluation Loop Start
    model.eval()

    losses = []

    for images, targets in val_loader:
        imgages, targets = map(lambda x: x.to(device), [images, targets])

        outputs = model(images)
        loss = criterion(outputs, targets)

        losses.append(loss.item())

    val_loss = sum(losses) / max(1, len(losses))
    writer.add_scalar('val_loss', val_loss, epoch)

    print(f"Validation: Loss={val_loss:.2f}")
    print("-"*25)

# model_path = '../fs-models/'
# os.makedirs(model_path, exist_ok=True)
# torch.save(model.state_dict(), model_path + 'resnet_18_multilabel.pth')
