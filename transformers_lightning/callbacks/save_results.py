import os
import csv
from argparse import ArgumentParser
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Sequence, Tuple, Union)

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn


class SaveTestResultsCallback(Callback):

    """
    Base class for terminal experiment loggers of evaluation results.

    Args:
        hparams:
            NameSpace with running arguments.
        agg_key_funcs:
            Dictionary which maps a metric name to a function, which will
            aggregate the metric values for the same steps.
        agg_default_func:
            Default function to aggregate metric values. If some metric name
            is not presented in the `agg_key_funcs` dictionary, then the
            `agg_default_func` will be used for aggregation.

    Note:
        The `agg_key_funcs` and `agg_default_func` arguments are used only when
        one logs metrics with the :meth:`~LightningLoggerBase.agg_and_log_metrics` method.
    """

    def __init__(self, hparams, *args, **kwargs):
        self.hparams = hparams
        super().__init__(*args, **kwargs)
        self.destination = os.path.join(self.hparams.output_dir, self.hparams.results_dir, self.hparams.name)

        raise ValueError("This class must be updated to the latest lightning release")

    @staticmethod
    def get_value(variable: Any) -> Any:
        """ Get the value of a tensor if it has size 1. """
        if isinstance(variable, torch.Tensor) and variable.numel() == 1:
            return variable.item()
        elif isinstance(variable, torch.Tensor):
            return variable.tolist()
        return variable

    def write(self, filename: str, data: dict):
        """ Write results to file """
        with open(os.path.join(self.destination, filename), "w") as fo:
            writer = csv.writer(fo, delimiter="\t", quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            # write keys as header
            writer.writerow(data.keys())
            # write values
            to_write = [self.__class__.get_value(x) for x in data.values()]
            for row in zip(*to_write):
                writer.writerow(row)

    @rank_zero_only
    def on_validation_start(self, trainer, pl_module):
        """Called when the trainer initialization begins, model has not yet been set."""
        # only run on main process
        if trainer.global_rank != 0:
            return

        os.makedirs(self.destination, exist_ok=True)

    @rank_zero_only
    def on_test_start(self, trainer, pl_module):
        """Called when the trainer initialization begins, model has not yet been set."""
        # only run on main process
        if trainer.global_rank != 0:
            return

        os.makedirs(self.destination, exist_ok=True)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        """Called when the validation loop ends."""
        # only run on main process
        if trainer.global_rank != 0:
            return

        metrics = trainer.callback_metrics

        if "results" in metrics:
            epoch = trainer.current_epoch
            step = pl_module.global_step
            for key, value in metrics['results'].items():
                filename = f"val_{key}-{epoch}-{step}-results.tsv"
                self.write(filename, value)

    @rank_zero_only
    def on_test_end(self, trainer, pl_module):
        """Called when the test ends."""
        # only run on main process
        if trainer.global_rank != 0:
            return

        metrics = trainer.callback_metrics

        if "results" in metrics:
            for key, value in metrics['results'].items():
                filename = f"test_{key}-results.tsv"
                self.write(filename, value)

    @staticmethod
    def add_callback_specific_args(parser: ArgumentParser):
        """ Add callback_specific arguments to parser. """
        # parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument('--paramenter_name', ...
        return parser
