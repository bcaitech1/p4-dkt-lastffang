from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR,ExponentialLR,CosineAnnealingWarmRestarts
from transformers import get_linear_schedule_with_warmup


def get_scheduler(optimizer, args):
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, patience=args.plateau_patience, factor=args.plateau_factor, mode='max', verbose=True)
    elif args.scheduler == 'linear_warmup':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps= args.warmup_steps,num_training_steps=args.total_steps)
    elif args.scheduler == 'step_lr':
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'exp_lr':
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    elif args.scheduler == 'cosine_annealing':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=args.eta_min)
    elif args.scheduler == 'cosine_annealing_warmstart':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=1, eta_min=args.eta_min, last_epoch=-1)

    return scheduler