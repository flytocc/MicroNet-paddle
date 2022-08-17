import math
import os
import sys
from typing import Iterable, Optional

import paddle
import paddle.amp.auto_cast as amp_cast
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.metric import accuracy

import util.lr_sched as lr_sched
import util.misc as misc
from util.data import Mixup
from util.model_ema import ModelEma


def clear_grad_(optimizer: optim.Optimizer):
    if isinstance(optimizer, paddle.fluid.optimizer.Optimizer):
        optimizer.clear_gradients()
    else:
        optimizer.clear_grad()


def train_one_epoch(model: nn.Layer, criterion: nn.Layer,
                    data_loader: Iterable, optimizer: optim.Optimizer,
                    epoch: int, loss_scaler,
                    model_ema: Optional[ModelEma] = None,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    num_training_steps_per_epoch=None, amp=False,
                    args=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ", log_file=os.path.join(args.output, "train.log"))
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = args.log_interval
    update_freq = args.update_freq

    clear_grad_(optimizer)

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if num_training_steps_per_epoch is not None and data_iter_step // update_freq >= num_training_steps_per_epoch:
            continue

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % update_freq == 0:
            lr = lr_sched.adjust_learning_rate(
                data_iter_step // update_freq / num_training_steps_per_epoch + epoch, args)
            optimizer.set_lr(lr)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with amp_cast(enable=amp):
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= update_freq
        if amp:
            norm = loss_scaler(loss, optimizer, parameters=model.parameters(),
                               update_grad=(data_iter_step + 1) % update_freq == 0)
            scale = loss_scaler.state_dict().get('scale')
        else:
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()

        if (data_iter_step + 1) % update_freq == 0:
            clear_grad_(optimizer)
            if model_ema is not None:
                model_ema.update(model)

        paddle.device.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % update_freq == 0:
            metrics = {'loss': loss_value_reduce, 'lr': lr}
            if amp:
                metrics.update({'norm': norm, 'scale': scale})
            log_writer.update(metrics)
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@paddle.no_grad()
def evaluate(data_loader, model, amp=False):
    criterion = nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        # compute output
        with amp_cast(enable=amp):
            output = model(images)
            loss = criterion(output, target)

        acc1 = accuracy(output, target.unsqueeze(-1), k=1)
        acc5 = accuracy(output, target.unsqueeze(-1), k=5)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
