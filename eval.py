import argparse

import paddle
from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler

import util.misc as misc
from util.datasets import build_dataset

from engine import evaluate

import models


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
parser.add_argument('--input_size', type=int, default=224, metavar='N',
                    help='Image patch size (default: 224)')
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
parser.add_argument('--resume', default='',
                    help='resume from checkpoint')
parser.add_argument('--dist_eval', action='store_true', default=False,
                    help='Enabling distributed evaluation')


def main(args):
    misc.init_distributed_mode(args)

    print("{}".format(args).replace(', ', ',\n'))

    dataset_val = build_dataset(is_train=False, args=args)

    if args.dist_eval:
        num_tasks = misc.get_world_size()
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        sampler_val = DistributedBatchSampler(
            dataset_val, args.batch_size, shuffle=True, drop_last=False)  # shuffle=True to reduce monitor bias
    else:
        sampler_val = BatchSampler(dataset=dataset_val, batch_size=args.batch_size)

    data_loader_val = DataLoader(dataset_val, batch_sampler=sampler_val, num_workers=args.num_workers)

    model = models.__dict__[args.model](
        num_classes=args.num_classes)

    model = paddle.DataParallel(model)
    model_without_ddp = model._layers
    n_parameters = sum(p.numel().item() for p in model.parameters() if not p.stop_gradient)
    print(f'number of params: {n_parameters / 1e6} M')

    misc.load_model(args, model_without_ddp)

    test_stats = evaluate(data_loader_val, model)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']}%")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
