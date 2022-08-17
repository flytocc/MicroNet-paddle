import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import yaml

import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler

import util.misc as misc
from util.data import Mixup
from util.datasets import build_dataset
from util.loss import LabelSmoothingCrossEntropy
from util.model_ema import ModelEma, unwrap_model
from engine import evaluate, train_one_epoch

import models

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside of the dataset group because it is positional.
parser.add_argument('data_dir', metavar='DIR',
                    help='path to dataset')
group.add_argument('--train_split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
group.add_argument('--val_split', metavar='NAME', default='val',
                    help='dataset validation split (default: val)')
group.add_argument('--cls_label_path_train', default=None, type=str,
                    help='dataset label path train (default: None)')
group.add_argument('--cls_label_path_val', default=None, type=str,
                    help='dataset label path val (default: None)')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default=None, type=str, metavar='MODEL',
                    help='Name of model to train (default: None')
group.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('--num_classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
group.add_argument('--input_size', type=int, default=224, metavar='N',
                    help='Image patch size (default: 224)')
group.add_argument('--crop_pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of dataset')
group.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
group.add_argument('-b', '--batch_size', type=int, default=128, metavar='N',
                    help='Input batch size for training (default: 128). Effective batch size is batch_size * update_freq * gpus')
group.add_argument('-vb', '--validation_batch_size', type=int, default=None, metavar='N',
                    help='Validation batch size override (default: None)')

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER', choices=['sgd', 'adam', 'adamw'],
                    help='Optimizer (default: "sgd"')
group.add_argument('--opt_eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
group.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
group.add_argument('--weight_decay', type=float, default=2e-5,
                    help='weight decay (default: 2e-5)')
group.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
parser.add_argument('--update_freq', default=1, type=int,
                    help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
group.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate (default: 0.05)')
group.add_argument('--warmup_lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
group.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
group.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
group.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
group.add_argument('--warmup_epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
group.add_argument('--cooldown_epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
group.add_argument('--t_in_epochs', action='store_true', default=False,
                    help='adjust lr per epoch')

# Augmentation & regularization parameters
group = parser.add_argument_group('Augmentation and regularization parameters')
group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
group.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
group.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
group.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
group.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                    help='Color jitter factor (default: None)')
group.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
group.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
group.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
group.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
group.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
group.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
group.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
group.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
group.add_argument('--mixup_prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
group.add_argument('--mixup_switch_prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
group.add_argument('--mixup_mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
group.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')

# Model Exponential Moving Average
group = parser.add_argument_group('Model exponential moving average parameters')
group.add_argument('--model_ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
group.add_argument('--model_ema_decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
group.add_argument('--log_interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
group.add_argument('-j', '--num_workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
group.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
group.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
group.add_argument('--log_wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')
group.add_argument('--wandb_project', default=None, type=str,
                    help='wandb project name')
group.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
group.add_argument('--dist_eval', action='store_true',
                    help='Enabling distributed evaluation')
group.add_argument('--model_ema_eval', action='store_true',
                    help='Using ema to eval during training.')
group.add_argument('--debug', action='store_true')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main(args):
    misc.init_distributed_mode(args)

    if misc.get_rank() == 0 and args.log_wandb and not args.eval:
        log_writer = misc.WandbLogger(args, entity=args.wandb_entity, project=args.wandb_project)
    else:
        log_writer = None

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    paddle.seed(args.seed)
    np.random.seed(seed)
    random.seed(seed)
    if args.debug:
        args.train_split = args.val_split
        paddle.version.cudnn.FLAGS_cudnn_deterministic = True

    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    validation_batch_size = args.validation_batch_size or args.batch_size
    sampler_train = DistributedBatchSampler(dataset_train, args.batch_size, shuffle=not args.debug, drop_last=True)
    if args.dist_eval:
        num_tasks = misc.get_world_size()
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        sampler_val = DistributedBatchSampler(dataset_val, validation_batch_size)
    else:
        sampler_val = BatchSampler(dataset=dataset_val, batch_size=validation_batch_size)

    data_loader_train = DataLoader(dataset_train, batch_sampler=sampler_train, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, batch_sampler=sampler_val, num_workers=args.num_workers)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)

    model = models.__dict__[args.model](
        num_classes=args.num_classes)

    eff_batch_size = args.batch_size * args.update_freq * misc.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // eff_batch_size

    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.update_freq)
    print("effective batch size: %d" % eff_batch_size)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    optim_kwargs = dict(
        learning_rate=args.lr,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        grad_clip=nn.ClipGradByGlobalNorm(args.clip_grad) if args.clip_grad is not None else None)
    if args.opt == 'sgd':
        if args.momentum > 0:
            optimizer = optim.Momentum(momentum=args.momentum, **optim_kwargs)
        else:
            optimizer = optim.SGD(**optim_kwargs)
    elif args.opt == 'adamw':
        decay_skip = model.no_weight_decay() if hasattr(model, 'no_weight_decay') else set()
        # following timm: set wd as 0 for bias and norm layers
        decay_dict = {param.name: not (len(param.shape) == 1 or name.endswith(".bias") or name in decay_skip)
                      for name, param in model.named_parameters()}
        bete1, beta2 = args.opt_betas or (0.9, 0.999)
        optimizer = optim.AdamW(
            beta1=bete1, beta2=beta2,
            epsilon=args.opt_eps or 1e-8,
            apply_decay_param_fun=lambda n: decay_dict[n],
            **optim_kwargs)
    else:
        raise NotImplementedError
    loss_scaler = misc.NativeScalerWithGradNormCount() if args.amp else None

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(model, decay=args.model_ema_decay, resume='')

    model = paddle.DataParallel(model)
    model_without_ddp = model._layers
    n_parameters = sum(p.numel().item() for p in model.parameters() if not p.stop_gradient)
    print(f'number of params: {n_parameters / 1e6} M')

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = nn.CrossEntropyLoss(soft_label=True)
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    misc.load_model(args, model_without_ddp, model_ema=model_ema,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, amp=args.amp)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print(f"Start training for {args.epochs} epochs + {args.cooldown_epochs} cooldown epochs")
    start_time = time.time()
    max_accuracy = 0.0
    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0

    if args.resume:
        test_stats = evaluate(data_loader_val, model, amp=args.amp)
        max_accuracy = max(max_accuracy, test_stats['acc1'])
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        if args.model_ema and args.model_ema_eval:
            test_stats_ema = evaluate(data_loader_val, unwrap_model(model_ema), amp=args.amp)
            max_accuracy_ema = max(max_accuracy_ema, test_stats_ema['acc1'])
            print(f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%")

    for epoch in range(args.start_epoch, args.epochs + args.cooldown_epochs):
        data_loader_train.batch_sampler.set_epoch(epoch)

        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, epoch, loss_scaler,
            model_ema, mixup_fn,
            log_writer=log_writer,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            amp=args.amp,
            args=args
        )
        test_stats = evaluate(data_loader_val, model, amp=args.amp)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

        if args.output:
            misc.save_model(args, epoch, model_without_ddp,
                            model_ema=model_ema, optimizer=optimizer, loss_scaler=loss_scaler, tag='latest')
            if test_stats["acc1"] > max_accuracy:
                misc.save_model(args, epoch, model_without_ddp,
                                model_ema=model_ema, optimizer=optimizer, loss_scaler=loss_scaler, tag='best')
            if (epoch + 1) % 20 == 0 or epoch + 1 == args.epochs or epoch + 1 == args.epochs + args.cooldown_epochs:
                misc.save_model(args, epoch, model_without_ddp,
                                model_ema=model_ema, optimizer=optimizer, loss_scaler=loss_scaler)

        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'best_acc1': max_accuracy,
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        # repeat testing routines for EMA, if ema eval is turned on
        if args.model_ema and args.model_ema_eval:
            test_stats_ema = evaluate(data_loader_val, unwrap_model(model_ema), amp=args.amp)
            print(f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%")
            if args.output and test_stats_ema["acc1"] > max_accuracy_ema:
                misc.save_model(args, epoch, model_without_ddp,
                                model_ema=model_ema, optimizer=optimizer, loss_scaler=loss_scaler, tag='best-ema')
            max_accuracy_ema = max(max_accuracy_ema, test_stats_ema["acc1"])
            print(f'Max accuracy of the model EMA: {max_accuracy_ema:.2f}%')
            log_stats.update({**{f'test_{k}_ema': v for k, v in test_stats_ema.items()},
                             'best_acc1_ema': max_accuracy_ema})

        if args.output and misc.is_main_process():
            if log_writer is not None:
                log_writer.update(log_stats)
                log_writer.flush()
            with open(os.path.join(args.output, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args, args_text = _parse_args()
    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)
    main(args)
