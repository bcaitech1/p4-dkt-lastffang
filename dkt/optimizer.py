from torch.optim import Adam, AdamW
from adamp import AdamP

def get_optimizer(model, args):
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    elif args.optimizer == 'adamW':
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    elif args.optimizer == 'adamP':
        optimizer = AdamP(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # 모든 parameter들의 grad값을 0으로 초기화
    optimizer.zero_grad()
    
    return optimizer