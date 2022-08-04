"""
Description: Networks related to the few shot learning part.

Author: Arrykrishna Mootoovaloo
Date: August 2022
Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
Project: One/Few-Shot Learning for Galaxy Zoo
"""

import os
import torch
import torch.nn as nn
from src.network import MultiLabelNet

# our scripts and functions
import settings as st

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ml_backbone(modelname: str):
    """Returns the model (backbone) which outputs the embeddings for the image. This function is for the multilabel case only.

    Args:
        modelname (str): name of the trained model, for example, ml-models-2022-5-25/resnet_18_multilabel_29.pth
    """

    # full path to the model
    fullpath = os.path.join(st.MODEL_PATH, modelname)

    # load the model
    loaded_model = torch.load(fullpath)

    # create the model again
    model = MultiLabelNet(backbone="resnet18")
    model = nn.DataParallel(model, device_ids=[0])
    model.load_state_dict(loaded_model)
    model.to(device)

    # the backbone
    chopped_layer = nn.Sequential(list(model.children())[0].backbone)

    return chopped_layer


def ml_feature_extractor(model: torch.nn.modules, image: torch.Tensor) -> torch.Tensor:
    """Extract the embeddings from the trained model given an image.

    Args:
        model (torch.nn.modules): the backbone
        image (torch.Tensor): the image to be used in the model.

    Returns:
        torch.Tensor: the feature vector
    """

    features = model(image.to(DEVICE))

    return features
