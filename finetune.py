"""
Description: Finetuning the multilabel network and adapt it to the few shot
learning case.

Author: Arrykrishna Mootoovaloo
Date: August 2022
Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
Project: One/Few-Shot Learning for Galaxy Zoo
"""
import datetime
import torch
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
TRAIN_MODEL = False

# Backbone
fname = 'ml-models-2022-5-25/resnet_18_multilabel_29.pth'

# Dataloaders (training)
support_dataset_train = FewShotFineTuneData(support=True, nshot=NSHOT, train=True)
support_loader_train = DataLoader(dataset=support_dataset_train, batch_size=16, shuffle=True)

query_dataset_train = FewShotFineTuneData(support=False, nshot=NSHOT, train=True)
query_loader_train = DataLoader(dataset=query_dataset_train, batch_size=16, shuffle=True)

# Dataloaders (validation)
support_dataset_valid = FewShotFineTuneData(support=True, nshot=NSHOT, train=False)
support_loader_valid = DataLoader(dataset=support_dataset_train, batch_size=16, shuffle=True)

query_dataset_valid = FewShotFineTuneData(support=False, nshot=NSHOT, train=False)
query_loader_valid = DataLoader(dataset=query_dataset_train, batch_size=16, shuffle=True)

# get the model (backbone)
backbone = ml_backbone(fname)

# Full network
weights_norm = normalise_weights('fewshot', f'train/vectors_mean_{str(NSHOT)}')

# Training
quant = {'lr': 1E-5,
         'weight_decay': 1E-5,
         'coefficient': 0.1,
         'nepochs': 300,
         'criterion': nn.CrossEntropyLoss()}

if TRAIN_MODEL:

    model_ft = FineTuneNet(backbone, True, weights_norm)
    model_ft.to(st.DEVICE)

    loaders_training = {'support': support_loader_train,
                        'query': query_loader_train}

    loaders_validation = {'support': support_loader_valid,
                          'query': query_loader_valid}

    # train the model
    model = training_finetune(model_ft, loaders_training, loaders_validation, quant, save=True)

else:
    nepochs = str(quant['nepochs'])
    fullpath = st.MODEL_PATH + f'finetune_{TODAY}_{nepochs}.pth'
    loaded_model = torch.load(fullpath, map_location=st.DEVICE)

    # create the model again
    model = FineTuneNet(backbone, True, weights_norm)
    model.load_state_dict(loaded_model)
    model.to(st.DEVICE)

    # make predictions
    query_loader = DataLoader(dataset=query_dataset_train, batch_size=1, shuffle=False)
    combined = finetuning_predictions(model, query_loader, NSHOT, nepochs=quant['nepochs'], save=True, train=True)
