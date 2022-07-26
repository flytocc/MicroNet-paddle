import math


def adjust_learning_rate(epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if args.t_in_epochs:
        epoch = math.floor(epoch)
    if epoch >= args.epochs:  # cooldown epochs
        lr = args.min_lr
    elif epoch < args.warmup_epochs:
        lr = args.warmup_lr + (args.lr - args.warmup_lr) * \
            epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1. + math.cos(
            math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    return lr
