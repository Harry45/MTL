"""
Description: Finetuning the multilabel network and adapt it to the few shot
learning case.

Author: Arrykrishna Mootoovaloo
Date: August 2022
Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
Project: One/Few-Shot Learning for Galaxy Zoo
"""
import datetime
import torch.nn as nn
from torch.utils.data import DataLoader

# our scripts and functions
import settings as st
from src.fewshot import ml_backbone, normalise_weights
from src.fewshot import finetuning_predictions, training_finetune
from src.dataset import FewShotFineTuneData
from src.network import FineTuneNet


DATE = datetime.datetime.now()
TODAY = str(DATE.year) + '-' + str(DATE.month) + '-' + str(DATE.day)
NSHOT = 10


# Backbone
fname = 'ml-models-2022-5-25/resnet_18_multilabel_29.pth'
backbone = ml_backbone(fname)

# Dataloaders
support_dataset = FewShotFineTuneData(support=True, nshot=NSHOT)
support_loader = DataLoader(dataset=support_dataset, batch_size=32, shuffle=True)

query_dataset = FewShotFineTuneData(support=False, nshot=NSHOT)
query_loader = DataLoader(dataset=query_dataset, batch_size=32, shuffle=True)

# Full network
weights_norm = normalise_weights('fewshot', f'vectors_mean_{str(NSHOT)}')
model_ft = FineTuneNet(backbone, True, weights_norm)
model_ft.to(st.DEVICE)

# Training
quant = {'lr': 1E-4,
         'weight_decay': 1E-5,
         'coefficient': 0.1,
         'nepochs': 150,
         'criterion': nn.CrossEntropyLoss()}

loaders = {'support': support_loader,
           'query': query_loader}

# train the model
model = training_finetune(model_ft, loaders, quant, save=True)

# make predictions
query_loader = DataLoader(dataset=query_dataset, batch_size=1, shuffle=False)
combined = finetuning_predictions(model, query_loader, NSHOT, save=True)
