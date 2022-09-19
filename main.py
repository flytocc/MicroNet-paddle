import argparse
import datetime
import json
import os
import time
from pathlib import Path

import yaml

import paddle
import paddle.nn as nn

import util.misc as misc
from data import Mixup, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from data.loader import FastCollateMixup, create_loader
from util.optim import AdamW, Momentum, SGD
from util.datasets import build_dataset
from util.loss import LabelSmoothingCrossEntropy
from util.model_ema import unwrap_model
from util.model_ema import ExponentialMovingAverageV2 as ModelEma
from util.random import random_seed 
from engine import evaluate, train_one_epoch

from models import create_model

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='Paddle ImageNet Training')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside of the dataset group because it is positional.
parser.add_argument('data_dir', metavar='DIR',
                    help='path to dataset')
group.add_argument('--train_split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
group.add_argument('--val_split', metavar='NAME', default='val',
                    help='dataset validation split (default: val)')
group.add_argument('--cls_label_path_train', default='', type=str,
                    help='dataset train label path (default: none)')
group.add_argument('--cls_label_path_val', default='', type=str,
                    help='dataset vallabel path (default: none)')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='', type=str, metavar='MODEL',
                    help='Name of model to train (default: none')
group.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('--num_classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
# group.add_argument('--img_size', type=int, default=None, metavar='N',
#                     help='Image patch size (default: None => model default)')
group.add_argument('--input_size', default=[3, 224, 224], nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
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
group.add_argument('--use_nesterov', action='store_true',
                    help='Use nesterov for sgd optimizer')
group.add_argument('--no_filter_bias_and_bn', action='store_true', default=False,
                    help='do not filter bias and bn')
group.add_argument('--weight_decay', type=float, default=2e-5,
                    help='weight decay (default: 2e-5)')
group.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
group.add_argument('--clip_mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value")')
# group.add_argument('--layer_decay', type=float, default=None,
#                     help='layer-wise learning rate decay (default: None)')

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
parser.add_argument('--update_freq', default=1, type=int,
                    help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
# group.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
#                     help='LR scheduler (default: "step"')
group.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate (default: 0.05)')
group.add_argument('--warmup_lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
group.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
group.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
group.add_argument('--start_epoch', type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
group.add_argument('--warmup_epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
group.add_argument('--cooldown_epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
group.add_argument('--t_in_epochs', action='store_true', default=False,
                    help='adjust lr per epoch')

# Augmentation & regularization parameters
group = parser.add_argument_group('Augmentation and regularization parameters')
group.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
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
group.add_argument('--aug_repeats', type=float, default=0,
                    help='Number of augmentation repetitions (distributed training only) (default: 0)')
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
group.add_argument('--mixup_off_epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
group.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
# group.add_argument('--train_interpolation', type=str, default='random',
#                     help='Training interpolation (random, bilinear, bicubic default: "random")')
# group.add_argument('--drop', type=float, default=0.0, metavar='PCT',
#                     help='Dropout rate (default: 0.)')
# group.add_argument('--drop_path', type=float, default=None, metavar='PCT',
#                     help='Drop path rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
group = parser.add_argument_group('Batch norm parameters', 'Only works with gen_efficientnet based models currently.')
# group.add_argument('--bn_momentum', type=float, default=None,
#                     help='BatchNorm momentum override (if not None)')
# group.add_argument('--bn_eps', type=float, default=None,
#                     help='BatchNorm epsilon override (if not None)')
group.add_argument('--sync_bn', action='store_true',
                    help='Enable synchronized BatchNorm.')

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
group.add_argument('--worker_seeding', type=str, default='all',
                    help='worker seed mode (default: all)')
group.add_argument('--log_interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
# group.add_argument('--recovery_interval', type=int, default=0, metavar='N',
#                     help='how many batches to wait before writing recovery checkpoint')
# group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
#                     help='number of checkpoints to keep (default: 10)')
group.add_argument('-j', '--num_workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
# group.add_argument('--save_images', action='store_true', default=False,
#                     help='save images of input bathes every log interval for debugging')
group.add_argument('--amp', action='store_true', default=False,
                    help='use AMP for mixed precision training')
group.add_argument('--no_prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
group.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
group.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
group.add_argument('--use_multi_epochs_loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
group.add_argument('--log_wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')
group.add_argument('--wandb_project', default=None, type=str,
                    help='wandb project name')
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


def main():
    misc.init_distributed_mode()
    args, args_text = _parse_args()

    args.filter_bias_and_bn = not args.no_filter_bias_and_bn
    args.prefetcher = not args.no_prefetcher
    args.distributed = misc.is_dist_avail_and_initialized()
    args.world_size = misc.get_world_size()
    args.rank = misc.get_rank()  # global rank
    assert args.rank >= 0

    log_writer = None
    if args.rank == 0 and args.log_wandb:
        if misc.has_wandb:
            log_writer = misc.WandbLogger(args, project=args.wandb_project)
        else:
            print("You've requested to log metrics to wandb but package not found. "
                  "Metrics not being logged to wandb, try `pip install wandb`")

    random_seed(args.seed, args.rank)

    model = create_model(
        args.model,
        num_classes=args.num_classes)

    print(f'Model {args.model} created, param count:{sum(m.numel().item() for m in model.parameters())}')

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print('Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
              'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')


    # setup automatic mixed-precision (AMP) loss scaling and op casting
    loss_scaler = None
    if args.amp:
        loss_scaler = misc.NativeScalerWithGradNormCount()
        print('Using Paddle AMP. Training in mixed precision.')

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = ModelEma(model, decay=args.model_ema_decay)

    # setup distributed training
    model_without_ddp = model
    if args.distributed:
        model = paddle.DataParallel(model)
        model_without_ddp = model._layers

    apply_decay_param_fun = None
    if args.filter_bias_and_bn:
        # setup learning rate schedule and starting epoch
        decay_skip = model.no_weight_decay() if hasattr(model, 'no_weight_decay') else set()
        # following timm: set wd as 0 for bias and norm layers
        decay_dict = {param.name: not (len(param.shape) == 1 or name.endswith(".bias") or name in decay_skip)
                      for name, param in model.named_parameters()}
        apply_decay_param_fun = lambda n: decay_dict[n]
    clip_func = {'norm': nn.ClipGradByGlobalNorm, 'value': nn.ClipGradByValue}[args.clip_mode]
    optim_kwargs = dict(
        learning_rate=args.lr,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        grad_clip=clip_func(args.clip_grad) if args.clip_grad is not None else None,
        apply_decay_param_fun=apply_decay_param_fun)

    if args.opt == 'sgd':
        if args.momentum > 0 or args.use_nesterov:
            optimizer = Momentum(momentum=args.momentum, use_nesterov=args.use_nesterov, **optim_kwargs)
        else:
            optimizer = SGD(**optim_kwargs)
    elif args.opt == 'adamw':
        bete1, beta2 = args.opt_betas or (0.9, 0.999)
        optimizer = AdamW(
            beta1=bete1, beta2=beta2,
            epsilon=args.opt_eps or 1e-8,
            **optim_kwargs)
    else:
        raise NotImplementedError
    num_epochs = args.epochs + args.cooldown_epochs

    start_epoch = misc.load_model(args, model_without_ddp, model_ema=model_ema,
                                  optimizer=optimizer, loss_scaler=loss_scaler)

    # create the train and eval datasets
    dataset_train = build_dataset(is_train=True, args=args)
    dataset_eval = build_dataset(is_train=False, args=args)

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # create data loaders w/ augmentation pipeiine
    interpolation = args.interpolation or 'bicubic'
    mean = args.mean or IMAGENET_DEFAULT_MEAN
    std = args.std or IMAGENET_DEFAULT_STD
    loader_train = create_loader(
        dataset_train,
        input_size=args.input_size,
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        interpolation=interpolation,
        mean=mean,
        std=std,
        num_workers=args.num_workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=True,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=args.input_size,
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation='bicubic' if interpolation == 'random' else interpolation,
        mean=mean,
        std=std,
        num_workers=args.num_workers,
        distributed=args.distributed,
        crop_pct=args.crop_pct,
        pin_memory=True,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
    )

    # setup loss function
    if mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        train_loss_fn = nn.CrossEntropyLoss(soft_label=True)
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()

    # setup checkpoint saver and eval metric tracking
    max_accuracy = 0.0
    max_accuracy_ema = 0.0
    eff_batch_size = args.batch_size * args.update_freq * args.world_size
    num_training_steps_per_epoch = len(dataset_train) // eff_batch_size
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                args.model,
                str(args.input_size[-1])
            ])
        args.output = os.path.join(args.output if args.output else "./output/train", exp_name)
        Path(args.output).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(args.output, 'args.yaml'), 'w') as f:
            f.write(args_text)

    print(f"Start training for {args.epochs} epochs + {args.cooldown_epochs} cooldown epochs")
    start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        if args.distributed and hasattr(loader_train.batch_sampler, 'set_epoch'):
            loader_train.batch_sampler.set_epoch(epoch)

        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_metrics = train_one_epoch(
            model, train_loss_fn, loader_train,
            optimizer, epoch, loss_scaler,
            model_ema, mixup_fn,
            log_writer=log_writer,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            amp=args.amp,
            update_freq=args.update_freq,
            args=args
        )
        test_metrics = evaluate(loader_eval, model, amp=args.amp)
        print(f"Accuracy of the network on the {len(dataset_eval)} test images: {test_metrics['acc1'] * 100}%")

        if args.output:
            misc.save_model(args, epoch, model_without_ddp,
                            model_ema=model_ema, optimizer=optimizer, loss_scaler=loss_scaler, tag='latest')
            if test_metrics["acc1"] > max_accuracy:
                misc.save_model(args, epoch, model_without_ddp,
                                model_ema=model_ema, optimizer=optimizer, loss_scaler=loss_scaler, tag='best')
            if (epoch + 1) % 20 == 0 or epoch + 1 == args.epochs or epoch + 1 == args.epochs + args.cooldown_epochs:
                misc.save_model(args, epoch, model_without_ddp,
                                model_ema=model_ema, optimizer=optimizer, loss_scaler=loss_scaler)

        max_accuracy = max(max_accuracy, test_metrics["acc1"])
        print(f'Max accuracy: {max_accuracy * 100}%')

        log_stats = {**{f'train_{k}': v for k, v in train_metrics.items()},
                     **{f'test_{k}': v for k, v in test_metrics.items()},
                     'best_acc1': max_accuracy,
                     'epoch': epoch}

        # repeat testing routines for EMA, if ema eval is turned on
        if args.model_ema:
            test_stats_ema = evaluate(loader_eval, unwrap_model(model_ema), amp=args.amp)
            print(f"Accuracy of the model EMA on {len(dataset_eval)} test images: {test_stats_ema['acc1'] * 100}%")
            if args.output and test_stats_ema["acc1"] > max_accuracy_ema:
                misc.save_model(args, epoch, model_without_ddp,
                                model_ema=model_ema, optimizer=optimizer, loss_scaler=loss_scaler, tag='best-ema')
            max_accuracy_ema = max(max_accuracy_ema, test_stats_ema["acc1"])
            print(f'Max accuracy of the model EMA: {max_accuracy_ema * 100}%')
            log_stats.update({**{f'test_{k}_ema': v for k, v in test_stats_ema.items()},
                              'best_acc1_ema': max_accuracy_ema})

        if args.output and misc.is_main_process():
            if log_writer is not None:
                log_writer.update(log_stats)
                log_writer.flush()
            with open(os.path.join(args.output, "log.txt"), mode="a", encoding="utf-8") as f:
                f.writelines([json.dumps(log_stats), "\n"])

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    main()
