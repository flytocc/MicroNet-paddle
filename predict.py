import argparse
from PIL import Image

import paddle
import paddle.amp as amp
import paddle.nn.functional as F

import util.misc as misc
from util.datasets import build_transform

import models


parser = argparse.ArgumentParser(description='Paddle ImageNet training and evaluation script')
parser.add_argument('--infer_imgs', default='./demo/ILSVRC2012_val_00020010.JPEG', type=str,
                    help='dataset path')
parser.add_argument('--model', default=None, type=str, metavar='MODEL',
                    help='Name of model to train (default: None')
parser.add_argument('--num_classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--input_size', type=int, default=224, metavar='N',
                    help='Image patch size (default: 224)')
parser.add_argument('--crop_pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--resume', default='',
                    help='resume from checkpoint')


def main(args):
    print("{}".format(args).replace(', ', ',\n'))

    preprocess = build_transform(is_train=False, args=args)

    model = models.__dict__[args.model](
        num_classes=args.num_classes)

    misc.load_model(args, model)

    # switch to evaluation mode
    model.eval()

    infer_imgs = args.infer_imgs
    if isinstance(args.infer_imgs, str):
        infer_imgs = [args.infer_imgs]

    images = [Image.open(img).convert('RGB') for img in infer_imgs]
    images = paddle.stack([preprocess(img) for img in images], axis=0)

    # compute output
    with amp.auto_cast():
        output = model(images)

    class_map = {}
    with open('demo/imagenet1k_label_list.txt', 'r') as f:
        for line in f.readlines():
            cat_id, *name = line.split('\n')[0].split(' ')
            class_map[int(cat_id)] = ' '.join(name)

    preds = []
    for file_name, scores, class_ids in zip(infer_imgs, *F.softmax(output).topk(5, 1)):
        preds.append({
            'class_ids': class_ids.tolist(),
            'scores': scores.tolist(),
            'file_name': file_name,
            'label_names': [class_map[i] for i in class_ids.tolist()]
        })

    print(preds)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
