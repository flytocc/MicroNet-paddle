import os
import argparse

import paddle

import util.misc as misc

import models


parser = argparse.ArgumentParser(description='Paddle ImageNet evalution')
parser.add_argument('--model', default=None, type=str, metavar='MODEL',
                    help='Name of model to train (default: None')
parser.add_argument('--num_classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--input_size', type=int, default=224, metavar='N',
                    help='Image patch size (default: 224)')
parser.add_argument('--resume', default='',
                    help='resume from checkpoint')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')


def main(args):
    # model
    model = models.__dict__[args.model](
        num_classes=args.num_classes)

    misc.load_model(args, model)

    shape = [-1, 3, args.input_size, args.input_size]

    model.eval()
    model = paddle.jit.to_static(
        model,
        input_spec=[paddle.static.InputSpec(shape=shape, dtype='float32')])
    save_path = os.path.join(args.output, 'model')
    paddle.jit.save(model, save_path)
    print(f'Model is saved in {args.output}.') # model.pdiparams|info|model


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
