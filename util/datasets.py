import os
from PIL import Image

from data import create_transform, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from paddle.io import Dataset
import paddle.vision.datasets as datasets


class ImageNetDataset(Dataset):

    def __init__(
            self,
            image_root,
            cls_label_path,
            transform=None):
        self._img_root = image_root
        self._cls_path = cls_label_path
        self.transform = transform
        self._load_anno()

    def _load_anno(self):
        assert os.path.exists(self._cls_path)
        assert os.path.exists(self._img_root)
        images = []
        labels = []

        with open(self._cls_path) as fd:
            lines = fd.readlines()
            for l in lines:
                l = l.strip().split(" ")
                images.append(os.path.join(self._img_root, l[0]))
                labels.append(int(l[1]))
                assert os.path.exists(images[-1]), images[-1]

        self.samples = list(zip(images, labels))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = Image.open(path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)


class DatasetFolder(datasets.DatasetFolder):
    _repr_indent = 4

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        if hasattr(self, "transform") and self.transform is not None:
            body += [repr(self.transform)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)


def build_dataset(is_train, args):
    if is_train and hasattr(args, 'cls_label_path_train') and args.cls_label_path_train:
        dataset = ImageNetDataset(args.data_dir, args.cls_label_path_train)
    elif not is_train and hasattr(args, 'cls_label_path_val') and args.cls_label_path_val:
        dataset = ImageNetDataset(args.data_dir, args.cls_label_path_val)
    else:
        root = os.path.join(args.data_dir, args.train_split if is_train else args.val_split)
        dataset = DatasetFolder(root)

    return dataset


def build_transform(is_train, args):
    trans_parmas = dict(
        input_size=args.input_size,
        is_training=is_train,
        interpolation=args.interpolation or 'bicubic',
        mean=args.mean or IMAGENET_DEFAULT_MEAN,
        std=args.std or IMAGENET_DEFAULT_STD,
        crop_pct=args.crop_pct,
    )

    if is_train:
        trans_parmas.update(
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            scale=args.scale,
            ratio=args.ratio,
            hflip=args.hflip,
            vflip=args.vflip,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
        )
    else:
        if trans_parmas['interpolation'] == 'random':
            trans_parmas['interpolation'] = 'bicubic'

    return create_transform(**trans_parmas)
