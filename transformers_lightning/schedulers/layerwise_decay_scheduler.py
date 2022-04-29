# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Converted to PyTorch by Luca Di liello
"""Functions and classes related to optimization (weight updates).
Modified from the original BERT code to allow for having separate learning
rates for different layers of the network.
"""
import math
from argparse import ArgumentError, Namespace

import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_warn

from transformers_lightning.schedulers.super_scheduler import SuperScheduler


class PolynomialLayerwiseDecaySchedulerWithWarmup(SuperScheduler):
    r"""
    Create a polynomially decreasing scheduler.
    Conversion of `https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/polynomial_decay`.
    More informations about the default parameters can be found on the documentation of
    `_LRScheduler` in the `torch` project.

    If `layerwise_lr_decay_power` is different from 1.0, the learning rate over each group will
    be multiplied by a factor `f = layerwise_lr_decay_power^(max(depth) - depth)`. `depth` is a
    key that should be defined in every group of parameters, along with the usual `weight_decay`.

    Args:
        hyperparameters: (:class:`~argparse.Namespace`):
            Collection of training hyperparameters.
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.

    Args through CLI:
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
        verbose (:obj:`bool`, `optional`, defaults to False):
            If ``True``, prints a message to stdout for each update.
        end_learning_rate (:obj:`float`, `optional`, defaults to 0.0001):
            target learning rate in the last step/epoch.
        lr_decay_power (:obj:`float`, `optional`, defaults to 1.0):
            learning rate decay base for polynomial decay.
        layerwise_lr_decay_power (:obj:`float`, `optional`, defaults to 1.0):
            learning rate decay base for layerwise decay.
        scheduler_cycle (:obj:`bool`, `optional`, defaults to False):
            whether to extend decay steps when global step is higher.
        warmup_steps (:obj:`int`, `optional`, defaults to 0):
            number of steps of the warmup phase.

    Example:
        >>> scheduler = PolynomialLayerwiseDecaySchedulerWithWarmup(hyperparameters, optimizer)
    """

    def __init__(self, hyperparameters: Namespace, optimizer: torch.optim.Optimizer):
        super().__init__(hyperparameters, optimizer)

        if hyperparameters.end_learning_rate < 0:
            raise ValueError("Cannot define negative `end_learning_rate`")

        if hyperparameters.layerwise_lr_decay_power < 0:
            raise ValueError("Cannot define negative `layerwise_lr_decay_power`")

        if hyperparameters.warmup_steps < 0:
            raise ValueError("Cannot define negative `warmup_steps`")

        # retrieve depth for each params group
        self.depths = [group['depth'] for group in optimizer.param_groups]

    def _layerwise_decay(self, lrs):
        r"""
        Have lower learning rates for layers closer to the input.
        Requires that groups passed to the Optimizer are already sorted from the
        closer to the input to the closer the output.
        """
        return [
            lr * (self.hyperparameters.layerwise_lr_decay_power**(max(self.depths) - depth))
            for lr, depth in zip(lrs, self.depths)
        ]

    def get_lr(self):
        if not self._get_lr_called_within_step:
            rank_zero_warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.", UserWarning
            )

        decay_steps = self.num_training_steps
        global_step = self.last_epoch
        # often last_epoch is used to indicate the number of steps. It is updated every `sched.step()`...

        # if scheduler_cycle, extend decay_steps if larger than global_step
        if self.hyperparameters.scheduler_cycle:
            decay_steps = decay_steps * math.ceil(global_step / decay_steps)
        else:
            global_step = min(global_step, decay_steps)

        lrs = [
            (
                (base_lr - self.hyperparameters.end_learning_rate) * (
                    1 - global_step / decay_steps
                ) ** (self.hyperparameters.lr_decay_power) + self.hyperparameters.end_learning_rate
            ) * (
                min(1.0, global_step / self.hyperparameters.warmup_steps)
                if self.hyperparameters.warmup_steps > 0 else 1.0
            ) for base_lr in self.base_lrs
        ]

        lrs = self._layerwise_decay(lrs)
        return lrs

    @staticmethod
    def add_scheduler_specific_args(parser: ArgumentError):
        r""" Add here the hyperparameters specific of the scheduler like the number of warmup steps. """
        super(
            PolynomialLayerwiseDecaySchedulerWithWarmup, PolynomialLayerwiseDecaySchedulerWithWarmup
        ).add_scheduler_specific_args(parser)
        parser.add_argument('--num_warmup_steps', type=int, default=0)
        parser.add_argument('--end_learning_rate', type=float, default=0.0001)
        parser.add_argument('--lr_decay_power', type=float, default=1.0)
        parser.add_argument('--layerwise_lr_decay_power', type=float, default=1.0)
        parser.add_argument('--scheduler_cycle', action='store_true')
