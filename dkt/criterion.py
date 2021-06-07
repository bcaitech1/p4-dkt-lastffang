import torch.nn as nn


def get_criterion(pred, target, args):
    if args.criterion == 'BCE':
        loss = nn.BCELoss(reduction="none")
    elif args.criterion == "BCELogit":
        loss = nn.BCEWithLogitsLoss(reduction="none")
    elif args.criterion == "MSE":
        loss = nn.MSELoss(reduction="none")
    elif args.criterion == "L1":
        loss = nn.L1Loss(reduction="none")
    # NLL, CrossEntropy not available
    else: # set default
        loss = nn.BCELoss(reduction="none")
    return loss(pred, target)