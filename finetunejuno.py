
"""
Description: Finetuning the multilabel network and adapt it to the few shot
learning case (JUNO dataset)

Author: Arrykrishna Mootoovaloo
Date: August 2022
Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
Project: One/Few-Shot Learning for Galaxy Zoo
"""
import datetime
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# our scripts and functions
import settings as st
from src.fewshot import ml_backbone, normalise_weights
from src.fewshot import finetuning_predictions
from src.dataset import FewShotFineTuneData
from src.network import FineTuneNet
from juno import finetuning, performance

DATE = datetime.datetime.now()
TODAY = str(DATE.year) + '-' + str(DATE.month) + '-' + str(DATE.day)
NSHOT = 10
_TRAIN_MODEL = False
_COEFFICIENT = 1E-3
_NEPOCHS = 300
_LR = 1E-5

parser = argparse.ArgumentParser(description='Training and prediction for the finetuning algorithm.')
parser.add_argument('--TRAIN_MODEL', help='Option to train the model',
                    default=_TRAIN_MODEL, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--COEFFICIENT', help='Regularisation factor', default=_COEFFICIENT, type=float)
parser.add_argument('--NEPOCHS', help='Number of epochs', default=_NEPOCHS, type=int)
parser.add_argument('--LR', help='Learning rate', default=_LR, type=float)
args = parser.parse_args()

# Backbone
fname = 'ml-models-2022-5-25/resnet_18_multilabel_29.pth'

# Dataloaders (training)
support_dataset_train = FewShotFineTuneData(support=True, nshot=NSHOT, train=True)
support_loader_train = DataLoader(dataset=support_dataset_train, batch_size=16, shuffle=True)

query_dataset_train = FewShotFineTuneData(support=False, nshot=NSHOT, train=True)
query_loader_train = DataLoader(dataset=query_dataset_train, batch_size=16, shuffle=True)


# get the model (backbone)
backbone = ml_backbone(fname)

# Full network
weights_norm = normalise_weights('juno', f'train/vectors_mean_{str(NSHOT)}')

# Training
quant = {'lr': args.LR,
         'weight_decay': 1E-5,
         'coefficient': args.COEFFICIENT,
         'nepochs': args.NEPOCHS,
         'criterion': nn.CrossEntropyLoss()}

if args.TRAIN_MODEL:

    model_ft = FineTuneNet(backbone, True, weights_norm)
    model_ft.to(st.DEVICE)

    loaders_training = {'support': support_loader_train,
                        'query': query_loader_train}

    # train the model
    model = finetuning(model_ft, loaders_training, quant, save=True)

else:
    nepochs = str(quant['nepochs'])
    fullpath = st.MODEL_PATH + f'finetune_juno_{TODAY}_{nepochs}.pth'
    loaded_model = torch.load(fullpath, map_location=st.DEVICE)

    # create the model again
    model = FineTuneNet(backbone, True, weights_norm)
    model.load_state_dict(loaded_model)
    model.to(st.DEVICE)

    # make predictions
    query_loader = DataLoader(dataset=query_dataset_train, batch_size=1, shuffle=False)
    combined = finetuning_predictions(model, query_loader, NSHOT, nepochs=quant['nepochs'], save=True, train=True)

    simple_acc = performance('juno', 'train/nearest_neighbour_10')
    fine_acc = performance('juno', f'finetune_train_10_{str(args.NEPOCHS)}')
    print(f'Accuracy with simple shot method : {simple_acc:.2f}')
    print(f'Accuracy with finetuning method  : {fine_acc:.2f}')
