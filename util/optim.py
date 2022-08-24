import paddle.optimizer as optim
from paddle import _C_ops
from paddle.fluid import core, framework
from paddle.fluid.framework import _in_legacy_dygraph, in_dygraph_mode
from paddle.fluid.regularizer import L2DecayRegularizer


class AdamW(optim.AdamW):
    r"""
    The AdamW optimizer
    """


class Momentum(optim.Momentum):

    def __init__(self, *args, apply_decay_param_fun=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._apply_decay_param_fun = apply_decay_param_fun

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)

        velocity_acc = self._get_accumulator(self._velocity_acc_str,
                                             param_and_grad[0])
        lr = self._create_param_lr(param_and_grad)

        # For fusion of momentum and l2decay 
        param = param_and_grad[0]
        regularization_method = self._regularization_method
        regularization_coeff = self._regularization_coeff
        if hasattr(param, 'regularizer'):
            # we skip param's l2decay before, so fuse it with momentum here.
            if isinstance(param.regularizer, L2DecayRegularizer):
                regularization_method = "l2_decay"
                regularization_coeff = param.regularizer._regularization_coeff
            # the param's regularization has been done before, we avoid do l2decay in momentum.
            elif param.regularizer is not None:
                regularization_method = ""
                regularization_coeff = 0.0

        #######################################################################

        # Whether we should do weight decay for the parameter.
        if self._apply_decay_param_fun is not None \
                and not self._apply_decay_param_fun(param_and_grad[0].name):
            regularization_method = ""
            regularization_coeff = 0.0

        #######################################################################

        find_master = self._multi_precision and param_and_grad[
            0].dtype == core.VarDesc.VarType.FP16
        master_weight = (self._master_weights[param_and_grad[0].name]
                         if find_master else None)

        if _in_legacy_dygraph():
            if isinstance(param_and_grad, dict):
                self._update_regularization(param_and_grad['weight_decay'])
            _, _, _ = _C_ops.momentum(
                param_and_grad[0], param_and_grad[1], velocity_acc, lr,
                master_weight, param_and_grad[0], velocity_acc, master_weight,
                'mu', self._momentum, 'use_nesterov', self._use_nesterov,
                'regularization_method', regularization_method,
                'regularization_coeff', regularization_coeff, 'multi_precision',
                find_master)
            return None
        if in_dygraph_mode():
            if isinstance(param_and_grad, dict):
                self._update_regularization(param_and_grad['weight_decay'])
            return _C_ops.final_state_momentum(
                param_and_grad[0], param_and_grad[1], velocity_acc, lr,
                master_weight, self._momentum, self._use_nesterov,
                regularization_method, regularization_coeff, find_master,
                self._rescale_grad)

        attrs = {
            "mu": self._momentum,
            "use_nesterov": self._use_nesterov,
            "regularization_method": regularization_method,
            "regularization_coeff": regularization_coeff,
            "multi_precision": find_master,
            "rescale_grad": self._rescale_grad
        }

        inputs = {
            "Param": [param_and_grad[0]],
            "Grad": [param_and_grad[1]],
            "Velocity": [velocity_acc],
            "LearningRate": [lr]
        }

        outputs = {
            "ParamOut": [param_and_grad[0]],
            "VelocityOut": [velocity_acc]
        }

        if find_master:
            inputs["MasterParam"] = master_weight
            outputs["MasterParamOut"] = master_weight

        # create the momentum optimize op
        momentum_op = block.append_op(
            type=self.type,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            stop_gradient=True)

        return momentum_op


class SGD(optim.SGD):

    def __init__(self, *args, apply_decay_param_fun=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._apply_decay_param_fun = apply_decay_param_fun

    def _create_regularization_of_grad(self, param, grad, regularization=None):
        # Whether we should do weight decay for the parameter.
        if self._apply_decay_param_fun is not None \
                and not self._apply_decay_param_fun(param.name):
            regularization = None

        return super()._create_regularization_of_grad(param, grad, regularization)
