import paddle
import paddle.vision.transforms as transforms
import paddle.vision.transforms.functional as F
from PIL import Image
import random
import numpy as np


class ToNumpy:

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ToTensor:

    def __init__(self, dtype=paddle.float32):
        self.dtype = dtype

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return paddle.to_tensor(np_img, dtype=self.dtype)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# Pillow is deprecating the top-level resampling attributes (e.g., Image.BILINEAR) in
# favor of the Image.Resampling enum. The top-level resampling attributes will be
# removed in Pillow 10.
if hasattr(Image, "Resampling"):
    _pil_interpolation_to_str = {
        Image.Resampling.NEAREST: 'nearest',
        Image.Resampling.BILINEAR: 'bilinear',
        Image.Resampling.BICUBIC: 'bicubic',
        Image.Resampling.BOX: 'box',
        Image.Resampling.HAMMING: 'hamming',
        Image.Resampling.LANCZOS: 'lanczos',
    }
else:
    _pil_interpolation_to_str = {
        Image.NEAREST: 'nearest',
        Image.BILINEAR: 'bilinear',
        Image.BICUBIC: 'bicubic',
        Image.BOX: 'box',
        Image.HAMMING: 'hamming',
        Image.LANCZOS: 'lanczos',
    }

_str_to_pil_interpolation = {b: a for a, b in _pil_interpolation_to_str.items()}


def str_to_pil_interp(mode_str):
    return _str_to_pil_interpolation[mode_str]


_RANDOM_INTERPOLATION = ('bilinear', 'bicubic')


class RandomResizedCropAndInterpolation(transforms.RandomResizedCrop):

    def _apply_image(self, img):
        interpolation = self.interpolation
        if self.interpolation == 'random':
            interpolation = random.choice(_RANDOM_INTERPOLATION)

        i, j, h, w = self._get_param(img)

        cropped_img = F.crop(img, i, j, h, w)
        return F.resize(cropped_img, self.size, interpolation)

    def __repr__(self):
        if self.interpolation == 'random':
            interpolate_str = f'({" ".join(_RANDOM_INTERPOLATION)})'
        else:
            interpolate_str = self.interpolation
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string
