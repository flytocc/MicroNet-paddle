# """ Loader Factory, Fast Collate, CUDA Prefetcher

# Prefetcher and Fast Collate inspired by NVIDIA APEX example at
# https://github.com/NVIDIA/apex/commit/d5e2bb4bdeedd27b1dfaf5bb2b24d6c000dee9be#diff-cf86c282ff7fba81fad27a559379d5bf

# Hacked together by / Copyright 2019, Ross Wightman
# """
import random
from functools import partial
from itertools import chain, repeat
from typing import Callable

import numpy as np

import paddle
from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler, IterableDataset, get_worker_info

from .transforms_factory import create_transform
from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .random_erasing import RandomErasing
from .mixup import FastCollateMixup


def fast_collate(batch):
    """ A fast collation function optimized for int32 images (np array or paddle) and int64 targets (labels)"""
    assert isinstance(batch[0], tuple)
    batch_size = len(batch)
    if isinstance(batch[0][0], tuple):
        # This branch 'deinterleaves' and flattens tuples of input tensors into one tensor ordered by position
        # such that all tuple of position n will end up in a paddle.split(tensor, batch_size) in nth position
        inner_tuple_size = len(batch[0][0])
        targets = paddle.to_tensor([b[1] for b in batch], dtype=paddle.int64)
        targets = paddle.tile(targets, [inner_tuple_size]).flatten()
        tensor = chain.from_iterable(zip(*[batch[i][0] for i in range(batch_size)]))
        tensor = np.stack(list(tensor), axis=0)
        tensor = paddle.to_tensor(tensor, dtype=paddle.uint8)
        return tensor, targets
    elif isinstance(batch[0][0], np.ndarray):
        targets = paddle.to_tensor([b[1] for b in batch], dtype=paddle.int64)
        assert len(targets) == batch_size
        tensor = np.stack([batch[i][0] for i in range(batch_size)], axis=0)
        tensor = paddle.to_tensor(tensor, dtype=paddle.uint8)
        return tensor, targets
    elif isinstance(batch[0][0], paddle.Tensor):
        targets = paddle.to_tensor([b[1] for b in batch], dtype=paddle.int64)
        assert len(targets) == batch_size
        tensor = paddle.stack([batch[i][0] for i in range(batch_size)], axis=0)
        return tensor, targets
    else:
        assert False


def expand_to_chs(x, n):
    if not isinstance(x, (tuple, list)):
        x = tuple(repeat(x, n))
    elif len(x) == 1:
        x = x * n
    else:
        assert len(x) == n, 'normalization stats must match image channels'
    return x


class PrefetchLoader:

    def __init__(
            self,
            loader: DataLoader,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            channels=3,
            fp16=False,
            re_prob=0.,
            re_mode='const',
            re_count=1,
            re_num_splits=0):

        mean = expand_to_chs(mean, channels)
        std = expand_to_chs(std, channels)
        normalization_shape = (1, channels, 1, 1)

        self.loader = loader
        self.mean = paddle.to_tensor([x * 255 for x in mean]).reshape(normalization_shape)
        self.std = paddle.to_tensor([x * 255 for x in std]).reshape(normalization_shape)
        self.fp16 = fp16
        if fp16:
            self.mean = self.mean.astype(paddle.float16)
            self.std = self.std.astype(paddle.float16)
        if re_prob > 0.:
            self.random_erasing = RandomErasing(
                probability=re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits)
        else:
            self.random_erasing = None

    def __iter__(self):
        # stream = paddle.device.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            # with paddle.device.cuda.stream_guard(stream):
            if self.fp16:
                next_input = paddle.to_tensor(next_input, dtype=paddle.float16)
            else:
                next_input = paddle.to_tensor(next_input, dtype=paddle.float32)
            next_input = (next_input - self.mean) / self.std
            if self.random_erasing is not None:
                next_input = self.random_erasing(next_input)

            if not first:
                yield input, target
            else:
                first = False

            # paddle.device.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __len__(self):
        return len(self.loader)

    @property
    def batch_sampler(self):
        return self.loader.batch_sampler

    @property
    def dataset(self):
        return self.loader.dataset

    @property
    def mixup_enabled(self):
        if isinstance(self.loader.collate_fn, FastCollateMixup):
            return self.loader.collate_fn.mixup_enabled
        else:
            return False

    @mixup_enabled.setter
    def mixup_enabled(self, x):
        if isinstance(self.loader.collate_fn, FastCollateMixup):
            self.loader.collate_fn.mixup_enabled = x


def _worker_init(worker_id, worker_seeding='all'):
    worker_info = get_worker_info()
    assert worker_info.id == worker_id
    if isinstance(worker_seeding, Callable):
        seed = worker_seeding(worker_info)
        random.seed(seed)
        paddle.seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
    else:
        assert worker_seeding in ('all', 'part')
        # random / paddle seed already called in dataloader iter class w/ worker_info.seed
        # to reproduce some old results (same seed + hparam combo), partial seeding is required (skip numpy re-seed)
        if worker_seeding == 'all':
            np.random.seed(worker_info.seed % (2 ** 32 - 1))


def create_loader(
        dataset,
        input_size,
        batch_size,
        is_training=False,
        use_prefetcher=True,
        no_aug=False,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_split=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        num_aug_repeats=0,
        num_aug_splits=0,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        distributed=False,
        crop_pct=None,
        collate_fn=None,
        pin_memory=False,
        fp16=False,
        tf_preprocessing=False,
        use_multi_epochs_loader=False,
        persistent_workers=True,
        worker_seeding='all',
):
    re_num_splits = 0
    if re_split:
        # apply RE to second half of batch if no aug split otherwise line up with aug split
        re_num_splits = num_aug_splits or 2
    dataset.transform = create_transform(
        input_size,
        is_training=is_training,
        use_prefetcher=use_prefetcher,
        no_aug=no_aug,
        scale=scale,
        ratio=ratio,
        hflip=hflip,
        vflip=vflip,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        tf_preprocessing=tf_preprocessing,
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
        re_num_splits=re_num_splits,
        separate=num_aug_splits > 0,
    )

    if distributed and not isinstance(dataset, IterableDataset):
        if is_training:
            if num_aug_repeats:
                raise NotImplementedError()
            else:
                batch_sampler = DistributedBatchSampler(dataset, batch_size, shuffle=True, drop_last=True)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            batch_sampler = DistributedBatchSampler(dataset, batch_size, shuffle=False, drop_last=False)
    else:
        assert num_aug_repeats == 0, "RepeatAugment not currently supported in non-distributed or IterableDataset use"
        if is_training:
            batch_sampler = BatchSampler(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            batch_sampler = BatchSampler(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    if collate_fn is None:
        collate_fn = fast_collate if use_prefetcher else None

    loader_class = DataLoader
    if use_multi_epochs_loader:
        loader_class = MultiEpochsDataLoader

    loader_args = dict(
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        use_shared_memory=pin_memory,
        worker_init_fn=partial(_worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    loader = loader_class(dataset, **loader_args)
    if use_prefetcher:
        prefetch_re_prob = re_prob if is_training and not no_aug else 0.
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            channels=input_size[0],
            fp16=fp16,
            re_prob=prefetch_re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits
        )

    return loader


class MultiEpochsDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

    def set_epoch(self, epoch):
        self.sampler.epoch = epoch
