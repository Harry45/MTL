"""
Description: Network for the Galaxy Zoo project.
"""

# Author: Arrykrishna Mootoovaloo
# Date: April 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Project: Multi-Task Learning for Galaxy Zoo

import torch
import torch.nn as nn
from torchvision import models

# our scripts and functions
import settings as st


class MultiLabelNet(nn.Module):
    """Initialise the neural network.

    Args:
        backbone (str, optional): The choice of the deep learning architecture. Defaults to "resnet18".

    Raises:
        ValueError: If the backbone is not supported.
    """

    def __init__(self, backbone="resnet18"):
        super(MultiLabelNet, self).__init__()

        if backbone not in models.__dict__:
            raise ValueError("Backbone {} not found in torchvision.models".format(backbone))

        # Create a backbone network from the pretrained models provided in torchvision.models
        self.backbone = models.__dict__[backbone](pretrained=True, progress=True)

        # change the first layer because we are using grayscale images
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # number of features as output by the network
        num_embedding = list(self.backbone.modules())[-1].out_features

        # add a few extra layers
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_embedding, st.NCLASS)
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.

        Args:
            img (torch.Tensor): The image to be passed through the encoder.

        Returns:
            torch.Tensor: The output of the encoder of shape [1, 1000] for a
            single image if we are using ResNet-18.
        """

        features = self.backbone(img)
        features = self.head(features)

        return features


class Encoder(nn.Module):
    """Initialise the encoder.

    Args:
        backbone (str, optional): The choice of the deep learning architecture. Defaults to "resnet18".

    Raises:
        ValueError: If the backbone is not supported.
    """

    def __init__(self, backbone="resnet18"):
        super(Encoder, self).__init__()

        if backbone not in models.__dict__:
            raise ValueError("Backbone {} not found in torchvision.models".format(backbone))

        # Create a backbone network from the pretrained models provided in torchvision.models
        self.encoder = models.__dict__[backbone](pretrained=False, progress=True)

        # change the first layer because we are using grayscale images
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.

        Args:
            img (torch.Tensor): The image to be passed through the encoder.

        Returns:
            torch.Tensor: The output of the encoder of shape [1, 1000] for a
            single image if we are using ResNet-18.
        """

        features = self.backbone(img)

        return features


class Decoder(nn.Module):
    """Initialise the decoder.

    Args:
        input_size (int): The input size of the decoder.
        output_size (int, optional): The output size of the decoder. Defaults to 1024.
    """

    def __init__(self, input_size: int, output_size: int = 1024):
        super(Decoder, self).__init__()

        # a custom neural network architecture - to implement one which has convolution
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

    def forward(self, embedding_vector: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder.

        Args:
            embedding_vector (torch.Tensor): The embedding vector to be passed through the decoder.

        Returns:
            torch.Tensor: The output of the decoder.
        """
        features = self.layers(embedding_vector)

        return features
