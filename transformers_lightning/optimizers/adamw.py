from argparse import ArgumentParser, Namespace
from typing import Generator

from torch.optim import AdamW

from transformers_lightning.optimizers.super_optimizer import SuperOptimizer
from transformers_lightning.optimizers.utils import get_parameters_grouped_for_weight_decay


class AdamWOptimizer(SuperOptimizer, AdamW):

    def __init__(self, hyperparameters: Namespace, named_parameters: Generator):
        r""" First hyperparameters argument to SuperOptimizer, other args for AdamW. """
        grouped_parameters = get_parameters_grouped_for_weight_decay(
            named_parameters, weight_decay=hyperparameters.weight_decay
        )
        super().__init__(
            hyperparameters,
            grouped_parameters,
            lr=hyperparameters.learning_rate,
            eps=hyperparameters.adam_epsilon,
            betas=hyperparameters.adam_betas
        )

    def add_optimizer_specific_args(parser: ArgumentParser):
        super(AdamWOptimizer, AdamWOptimizer).add_optimizer_specific_args(parser)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--adam_epsilon', type=float, default=1e-8)
        parser.add_argument('--adam_betas', nargs=2, type=float, default=[0.9, 0.999])
