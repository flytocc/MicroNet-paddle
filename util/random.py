import random
import numpy as np
import paddle


def random_seed(seed=42, rank=0):
    paddle.seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
