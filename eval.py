import argparse

import paddle
from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler

import util.misc as misc
from util.datasets import build_dataset
from data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from data.loader import create_loader

from engine import evaluate

from models import create_model


parser = argparse.ArgumentParser(description='Paddle ImageNet evalution')
parser.add_argument('data_dir', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--val_split', metavar='NAME', default='val',
                    help='dataset split (default: validation)')
parser.add_argument('--model', default=None, type=str, metavar='MODEL',
                    help='Name of model to train (default: None')
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--input_size', default=[3, 224, 224], nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop_pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--num_classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--cls_label_path_val', default=None, type=str,
                    help='dataset label path')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use AMP for mixed precision training')
parser.add_argument('--no_prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--resume', default='',
                    help='resume from checkpoint')


def main():
    misc.init_distributed_mode()
    args = parser.parse_args()

    args.prefetcher = not args.no_prefetcher
    args.distributed = misc.is_dist_avail_and_initialized()

    dataset_eval = build_dataset(is_train=False, args=args)

    # create data loaders w/ augmentation pipeiine
    interpolation = args.interpolation or 'bicubic'
    loader_eval = create_loader(
        dataset_eval,
        input_size=args.input_size,
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation='bicubic' if interpolation == 'random' else interpolation,
        mean=args.mean or IMAGENET_DEFAULT_MEAN,
        std=args.std or IMAGENET_DEFAULT_STD,
        num_workers=args.num_workers,
        distributed=args.distributed,
        crop_pct=args.crop_pct,
        pin_memory=True,
    )

    model = create_model(
        args.model,
        num_classes=args.num_classes)
    print(f'Model {args.model} created, param count:{sum(m.numel().item() for m in model.parameters())}')

    misc.load_model(args, model)

    if args.distributed:
        model = paddle.DataParallel(model)

    test_stats = evaluate(loader_eval, model, args.amp)
    print(f"Accuracy of the network on the {len(loader_eval)} test images: {test_stats['acc1'] * 100}%")


if __name__ == '__main__':
    main()
