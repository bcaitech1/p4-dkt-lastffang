
import torch.nn as nn


def get_criterion(pred, target, args):
    if args.criterion == 'BCE':
        loss = nn.BCELoss(reduction="none")
    return loss(pred, target)