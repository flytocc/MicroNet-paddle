from copy import deepcopy

import paddle
import paddle.nn as nn


class ExponentialMovingAverage:
    """
    Exponential Moving Average
    Code was heavily based on https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/model_ema.py
    """

    def __init__(self, model: nn.Layer, decay=0.9998):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    @paddle.no_grad()
    def _update(self, model, update_fn):
        for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
            ema_v.set_value(update_fn(ema_v.numpy(), model_v.numpy()))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def set_state_dict(self, *args, **kwargs):
        return self.module.set_state_dict(*args, **kwargs)


class ExponentialMovingAverageV2(ExponentialMovingAverage):

    def __init__(self, model: nn.Layer, decay=0.9998):
        super().__init__(model, decay)
        self._update_weights()

    @paddle.no_grad()
    def _update(self, model, update_fn):
        for name, model_v in model.state_dict().items():
            self._weights[name] = update_fn(self._weights[name], model_v.numpy())

    def _update_weights(self):
        self._weights = {name: deepcopy(param).numpy() for name, param in self.module.state_dict().items()}

    def _update_module(self):
        for name, ema_v in self.module.state_dict().items():
            ema_v.set_value(self._weights[name])

    def state_dict(self, *args, **kwargs):
        self._update_module()
        return self.module.state_dict(*args, **kwargs)

    def set_state_dict(self, *args, **kwargs):
        ret = self.module.set_state_dict(*args, **kwargs)
        self._update_weights()
        return ret


def unwrap_model(model):
    if isinstance(model, ExponentialMovingAverageV2):
        model._update_module()
        return model.module
    return model.module if hasattr(model, 'module') else model
