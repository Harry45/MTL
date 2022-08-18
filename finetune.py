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
from src.fewshot import ml_backbone, normalise_weights, shanon_entropy
from src.fewshot import finetuning_predictions  # training_finetune
from src.dataset import FewShotFineTuneData
from src.network import FineTuneNet
import utils.helpers as hp


DATE = datetime.datetime.now()
TODAY = str(DATE.year) + '-' + str(DATE.month) + '-' + str(DATE.day)
TIME = str(DATE.hour) + ':' + str(DATE.minute)


NSHOT = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def training_finetune(model: nn.Module, loaders: dict, quant: dict, save: bool) -> nn.Module:
    """Fine tune the model and adapt it to the few shot learning task.

    Args:
        model (nn.Module): the pre-trained deep learning model.
        loaders (dict): a dictionary for the support and query data.
        quant (dict): a dictionary with important quantities, for example,

        quant = {'lr': 1E-3,
        'weight_decay': 1E-5,
        'coefficient': 0.01,
        'nepochs': 150,
        'criterion': nn.CrossEntropyLoss()}

        where weight decay is a regularizer, coefficient is a regularization
        term for the Shanon Entropy calculation and so forth.

        save (bool): if True, the model will be save to the models directory

    Returns:
        nn.Module: the fine tuned model.
    """

    # choose an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=quant['lr'], weight_decay=quant['weight_decay'])

    # iterate over the whole dataset
    for epoch in range(quant['nepochs']):

        model.train()

        losses = []
        loss_support_rec = []
        loss_query_rec = []

        # calculate loss due to the support set
        for images, targets in loaders['support']:
            images, targets = map(lambda x: x.to(DEVICE), [images, targets])

            outputs = model(images)

            loss_support = quant['criterion'](outputs, targets.view(-1))

            optimizer.zero_grad()
            loss_support.backward()
            optimizer.step()

            loss_support_rec.append(loss_support.item())

        # calculate loss due to the query set (transductive learning)
        for images, targets in loaders['query']:
            images, targets = map(lambda x: x.to(DEVICE), [images, targets])

            outputs = model(images)

            loss_query = quant['coefficient'] * shanon_entropy(outputs)

            optimizer.zero_grad()
            loss_query.backward()
            optimizer.step()

            loss_query_rec.append(loss_query.item())

        # calculate the total loss
        total_loss = sum(loss_support_rec) / len(loss_support_rec) + sum(loss_query_rec) / len(loss_query_rec)

        losses.append(total_loss)

        print(f"Epoch [{epoch + 1} / {quant['nepochs']}]: Loss = {total_loss:.4f}")

    # save the values of the loss
    if save:
        hp.save_pickle(losses, 'results', f'finetune_{TODAY}_{TIME}')
        torch.save(model.state_dict(), st.MODEL_PATH + f'finetune_{TODAY}.pth')

    return model


# Backbone
fname = 'ml-models-2022-5-25/resnet_18_multilabel_29.pth'
backbone = ml_backbone(fname)

# Dataloaders
support_dataset = FewShotFineTuneData(support=True, nshot=NSHOT)
support_loader = DataLoader(dataset=support_dataset, batch_size=8, shuffle=True)

query_dataset = FewShotFineTuneData(support=False, nshot=NSHOT)
query_loader = DataLoader(dataset=query_dataset, batch_size=8, shuffle=True)

# Full network
weights_norm = normalise_weights('fewshot', f'vectors_mean_{str(NSHOT)}')
model_ft = FineTuneNet(backbone, True, weights_norm)
model_ft.to(DEVICE)

# Training
quant = {'lr': 1E-4,
         'weight_decay': 1E-5,
         'coefficient': 0.01,
         'nepochs': 150,
         'criterion': nn.CrossEntropyLoss()}

loaders = {'support': support_loader,
           'query': query_loader}

# train the model
model = training_finetune(model_ft, loaders, quant, save=True)

# make predictions
query_loader = DataLoader(dataset=query_dataset, batch_size=1, shuffle=False)
combined = finetuning_predictions(model, query_loader, NSHOT, save=True)
