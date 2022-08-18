"""
Description: Network for the Galaxy Zoo project.
"""

# Author: Arrykrishna Mootoovaloo
# Date: April 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Project: Multi-Task Learning for Galaxy Zoo

from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# our scripts and functions
import settings as st


class FineTuneNet(nn.Module):
    """Network for finetuning according to the few-shot learning.

    Args:
        backbone (nn.Module): a pre-trained network to be used as the backbone
        applyrelu (bool): if True, apply relu on the output of the backbone
        weightmatrix (torch.Tensor): the weight matrix to be used for the linear layer
    """

    def __init__(self, backbone: nn.Module, applyrelu: bool, weightmatrix: torch.Tensor):

        super().__init__()

        # make a copy of the backbone
        self.backbone = deepcopy(backbone)

        # set the backbone to evaluation mode
        self.backbone.eval()

        # number of output features
        self.numfeatures = list(backbone.modules())[-1].out_features

        self.applyrelu = applyrelu

        # create a head on top (4 because we have 4 classes)
        self.linear = nn.Linear(self.numfeatures, st.NWAYS, bias=False)

        with torch.no_grad():
            self.linear.weight.copy_(weightmatrix)

    def forward(self, xtensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output from the backbone of size 1 x 1000.
        """

        # get the embedding vectors
        xtensor = self.backbone(xtensor)

        # apply relu
        if self.applyrelu:
            xtensor = torch.relu(xtensor)

        # normalise the features
        xtensor = torch.nn.functional.normalize(xtensor)

        # apply the classifier on top now
        xtensor = self.linear(xtensor)

        return xtensor


class MultiLabelNet(nn.Module):
    """Initialise the neural network.

    Args:
        backbone (str, optional): The choice of the deep learning architecture. Defaults to "resnet18".

    Raises:
        ValueError: If the backbone is not supported.
    """

    def __init__(self, backbone="resnet18"):
        super(MultiLabelNet, self).__init__()

        # create the backbone network
        self.backbone = Encoder(backbone)

        # number of features as output by the network
        num_embedding = list(self.backbone.modules())[-1].out_features

        # add a few extra layers
        self.head = nn.Sequential(
            nn.Dropout(0.5),
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


class MultiTaskNet(nn.Module):
    """Neural Network architecture for the multi-task problem.

    Args:
        backbone (str, optional): The network architecture to use. Defaults to "resnet18".
        output_size (dict, optional): A dictionary of the output size. Defaults to None.

    Raises:
        ValueError: If the backbone is not supported.
    """

    def __init__(self, backbone="resnet18", output_size: dict = None, resnet_task: bool = True, kernel_size: int = 3):
        super(MultiTaskNet, self).__init__()

        # create the backbone network
        self.backbone = Encoder(backbone)

        # number of features as output by the network
        num_embedding = list(self.backbone.modules())[-1].out_features

        self.decoders = nn.ModuleDict()

        for i in range(st.NUM_TASKS):

            if resnet_task:
                self.decoders['task_' + str(i + 1)] = DecoderResNet(kernel_size, output_size['task_' + str(i + 1)])

            else:
                self.decoders['task_' + str(i + 1)] = Decoder(num_embedding, output_size['task_' + str(i + 1)])

    def forward(self, img: torch.Tensor) -> dict:
        """Forward pass through the encoder and decoders for each task.

        Args:
            img (torch.Tensor): The image to be passed through the encoder.

        Returns:
            dict: The output of the decoders for each task.
        """
        shared_feature = self.backbone(img)

        tasks = {}
        for i in range(st.NUM_TASKS):
            tasks['task_' + str(i + 1)] = self.decoders['task_' + str(i + 1)](shared_feature)

        return tasks


class DecoderResNet(nn.Module):
    """A ResNet implementation, applied to 1D features

    Args:
        kernel_size (int): The kernel size of the convolutional layers.
        output_size (int): The output size of the final layer.
    """

    def __init__(self, kernel_size: int, output_size: int):
        super(DecoderResNet, self).__init__()

        # record the output size
        self.output_size = output_size

        # kernel size
        self.kernel_size = kernel_size

        # padding
        self.padding = (self.kernel_size - 1) // 2

        # first convolutional layer
        self.conv1 = nn.Conv1d(1, 1, kernel_size=self.kernel_size, stride=2, padding=self.padding)

        # batch normalisation
        self.bn1 = nn.BatchNorm1d(1)

        # second convolutional layer
        self.conv2 = nn.Conv1d(1, 1, kernel_size=self.kernel_size, stride=2, padding=self.padding)

        # batch normalisation
        self.bn2 = nn.BatchNorm1d(1)

        # last layer
        # we needed the value 250 to build this sequential layer
        self.last_layer = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(250, self.output_size)
        )

    def forward(self, embedding_vector: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder.

        Args:
            embedding_vector (torch.Tensor): The embedding vector to be passed through the decoder.

        Returns:
            torch.Tensor: The output of the decoder.
        """
        embedding_vector = embedding_vector.unsqueeze(1)
        nfeatures = embedding_vector.shape[2]

        # first set of operations
        out = self.conv1(embedding_vector)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # we have to pad the output so we can add it to the input
        nout = out.shape[2]
        left = (nfeatures - nout) // 2
        right = nfeatures - nout - left
        out = F.pad(out, (left, right), mode='constant', value=0)

        # add the output to the input
        out += embedding_vector
        out = F.relu(out)

        # second set of operations
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # apply last layer
        out = out.view(out.shape[0], -1)
        out = self.last_layer(out)

        return out


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
            torch.Tensor: The output of the encoder is of shape [1, 1000] for a
            single image if we are using ResNet-18.
        """

        features = self.encoder(img)

        return features
