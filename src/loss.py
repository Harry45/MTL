"""
Description: Customised loss functions for the Galaxy Zoo project.
"""

# Author: Arrykrishna Mootoovaloo
# Date: April 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Project: Multi-Task Learning for Galaxy Zoo

import torch
import torch.nn as nn

# to assign weights to this loss function
LOSS_FUNC = nn.MultiLabelSoftMarginLoss(reduction='none')


def criterion_multi_label(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """The loss criterion to use for the multi-label classification. Since the
    decision tree for the Galaxy Zoo project contains NaNs, we modify the loss
    function so that there is no gradient flow where there is no labels.

    Args:
        predictions (torch.Tensor): the predictions from the network
        targets (torch.Tensor): the targets, for example, [0, 1, 1, 0] for an
        object with 4 labels

    Returns:
        torch.Tensor: the value of the loss
    """

    # labels which are greater or equal to zero
    gez = targets >= 0

    # calculate the loss
    loss = LOSS_FUNC(predictions[gez].view(1, -1), targets[gez].view(1, -1))

    return loss
