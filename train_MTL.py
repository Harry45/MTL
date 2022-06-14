"""
Description: Train the network for the multi-task case.
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
from src.network import MultiTaskNet
from src.dataset import DECaLSDataset
import settings as st

date = datetime.datetime.now()
today = str(date.year) + '-' + str(date.month) + '-' + str(date.day)

out_path = './output/'
model_path = '/data/phys-zooniverse/phys2286/Models/mtl-models-' + today + '/'

# make the folders
os.makedirs(out_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# Set device to CUDA if a CUDA device is available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create the dataloader
train_dataset = DECaLSDataset(mode='train', augment=False, multi_task=True)
val_dataset = DECaLSDataset(mode='validate', augment=False, multi_task=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

# define the model
model = MultiTaskNet(backbone="resnet18", output_size=st.LABELS_PER_TASK, resnet_task=True)
model = nn.DataParallel(model)
model.to(device)

# set the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1E-4, weight_decay=1E-5)
loss_func = nn.BCEWithLogitsLoss()


def criterion(loss_function, outputs: nn.ModuleDict, labels: dict) -> torch.tensor:
    losses = 0

    # we can add a weight for each loss
    # for example, we can calculate the fraction of objects which go into Nodes
    # 2, 3, 4 and so forth and then weight the loss accordingly
    for _, key in enumerate(outputs):
        losses += loss_function(outputs[key], labels[key].float().to(device))
    return losses


def mod_criterion(outputs: nn.ModuleDict, labels: dict) -> torch.tensor:
    losses = 0

    for _, key in enumerate(outputs):
        lossfunc = nn.BCEWithLogitsLoss(weight=st.WEIGHTS_MTL[key], reduction='mean')
        losses += lossfunc(outputs[key], labels[key].float().to(device))

    return losses


writer = SummaryWriter(os.path.join(out_path, "summary"))

epochs = 30

for epoch in range(epochs):
    print("Epoch [{} / {}]".format(epoch + 1, epochs))
    model.train()

    losses = []

    # Training Loop Start
    for images, targets in train_loader:

        images = images.to(device)

        outputs = model(images)
        # loss = criterion(loss_func, outputs, targets)
        loss = mod_criterion(outputs, targets)
        # loss = mtl(images, targets)

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
        images = images.to(device)

        # images, targets = map(lambda x: x.to(device), [images, targets])

        outputs = model(images)
        # loss = criterion(loss_func, outputs, targets)
        loss = mod_criterion(outputs, targets)
        # loss = mtl(images, targets)

        losses.append(loss.item())

    val_loss = sum(losses) / len(losses)
    writer.add_scalar('val_loss', val_loss, epoch)

    print(f"Validation : Loss={val_loss:.2e}")
    print("-" * 30)

    # torch.save(model.state_dict(), model_path + 'resnet_18_multitask_' + str(epoch) + '.pth')

# --------------------------------------------------------------
# class MultiTaskLoss(nn.Module):
#     def __init__(self, tasks):
#         super(MultiTaskLoss, self).__init__()
#         self.tasks = tasks
#         self.sigma = nn.Parameter(torch.ones(st.NUM_TASKS))
#         self.loss_func = nn.BCEWithLogitsLoss()

#     def forward(self, images, targets):
#         losses = []
#         outputs = self.tasks(images)

#         for _, key in enumerate(targets):

#             loss_per_task = self.loss_func(outputs[key], targets[key].float().to(device))
#             losses.append(loss_per_task)

#         losses = torch.Tensor(losses).to(device) / self.sigma**2
#         total_loss = losses.sum() + torch.log(self.sigma.prod())
#         print(self.sigma)
#         return total_loss


# mtl = MultiTaskLoss(model).to(device)
# optimizer = torch.optim.Adam(mtl.parameters(), lr=1E-4, weight_decay=1E-5)

# --------------------------------------------------------------
