import json
import logging
import os
from argparse import ArgumentParser, Namespace
from io import TextIOWrapper
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.logger import _add_prefix, _convert_params, _flatten_dict
from pytorch_lightning.utilities.logger import _sanitize_params as _utils_sanitize_params

logger = logging.getLogger(__name__)


class JsonBoardLogger(LightningLoggerBase):
    r"""
    Log to local file system in `JsonBoard <https://github.com/lucadiliello/jsonboard>`_ format.

    Logs are saved to ``os.path.join(save_dir, name, version)``.,

    Example:

    .. testcode::

        from pytorch_lightning import Trainer
        from transformers_lightning.loggers import JsonBoardLogger

        logger = JsonBoardLogger("js_logs", name="my_model")
        trainer = Trainer(logger=logger)

    Args:
        hyperparameters: Namespace with training hyperparameters.

    """

    NAME_OUTPUT_FILE = "data.jsonl"
    NAME_HPARAMS_FILE = "hparams.json"
    NAME_METADATA_FILE = "meta.json"

    def __init__(self, hyperparameters: Namespace):
        super().__init__()
        self.hyperparameters = hyperparameters
        self._name = hyperparameters.name
        self._version = None
        self._fs = get_filesystem(hyperparameters.jsonboard_dir)
        self._experiment = None
        self.hparams = {}
        self.meta = {}

    def reset(self):
        r""" Reset experiment. """
        self._experiment = None

    @property
    def root_dir(self) -> str:
        r""" Parent directory for all JsonBoard checkpoint subdirectories.

        If the experiment name parameter is ``None`` or the empty string, no experiment subdirectory is used and the
        checkpoint will be saved in "save_dir/version_dir"
        """
        return os.path.join(self.save_dir, self.hyperparameters.name)

    @property
    def log_dir(self) -> str:
        r""" The directory for this run's JsonBoard checkpoint.

        By default, it is named ``'version_${self.version}'``.
        """
        # create a pseudo standard paths
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, version)
        log_dir = os.path.expandvars(log_dir)
        log_dir = os.path.expanduser(log_dir)
        return log_dir

    @property
    def save_dir(self) -> Optional[str]:
        r""" Gets the save directory where the JsonBoard experiments are saved.

        Returns:
            The local path to the save directory where the JsonBoard experiments are saved.
        """
        return os.path.join(self.hyperparameters.output_dir, self.hyperparameters.jsonboard_dir)

    @property
    @rank_zero_experiment
    def experiment(self) -> TextIOWrapper:
        r"""
        Actual JsonBoard object. To use JsonBoard features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_tensorboard_function()

        """
        if self._experiment is not None:
            return self._experiment

        assert rank_zero_only.rank == 0, "tried to init log dirs in non global_rank=0"

        if self.log_dir:
            self._fs.makedirs(self.log_dir, exist_ok=True)

        filename = os.path.join(self.log_dir, self.NAME_OUTPUT_FILE)
        self._experiment = open(filename, "a")
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace] = None) -> None:
        r""" Record hyperparameters.

        Args:
            params: a dictionary-like container with the hyperparameters
        """

        assert rank_zero_only.rank == 0, "tried to init log dirs in non global_rank=0"

        params = _convert_params(params)

        # store params to output
        self.hparams.update(params)

        # format params into the suitable for tensorboard
        params = _flatten_dict(params)
        params = self._sanitize_params(params)

        if self.log_dir:
            self._fs.makedirs(self.log_dir, exist_ok=True)

        filename = os.path.join(self.log_dir, self.NAME_HPARAMS_FILE)
        if os.path.isfile(filename):
            # going to update existing file
            with open(filename, "r") as fh:
                params.update(json.load(fh))

        with open(filename, "w") as fh:
            json.dump(params, fh)

    @rank_zero_only
    def log_metadata(self, metadata: Union[Dict[str, Any], Namespace] = None) -> None:
        r""" Record metadata.

        Args:
            metadata: a dictionary-like container with the hyperparameters
        """

        assert rank_zero_only.rank == 0, "tried to init log dirs in non global_rank=0"

        metadata = _convert_params(metadata)

        # store params to output
        self.meta.update(metadata)

        # format params into the suitable for tensorboard
        metadata = _flatten_dict(metadata)
        metadata = self._sanitize_params(metadata)

        if self.log_dir:
            self._fs.makedirs(self.log_dir, exist_ok=True)

        filename = os.path.join(self.log_dir, self.NAME_METADATA_FILE)
        if os.path.isfile(filename):
            # going to update existing file
            with open(filename, "r") as fh:
                metadata.update(json.load(fh))

        with open(filename, "w") as fh:
            json.dump(metadata, fh)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        r""" Just write the metrics to disk. """
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"
        metrics = _add_prefix(metrics)
        if metrics:
            try:
                self._sanitize_and_write_metrics(metrics, step + 1)
            except TypeError as ex:
                raise ValueError(
                    f"\n you tried to log {metrics} which is not currently supported. Try a dict or a scalar/tensor."
                ) from ex

    @rank_zero_only
    def finalize(self, status: str):
        r""" First log eventually the last remained metrics and then close experiment. """
        self.save()
        self.experiment.flush()
        self.experiment.close()
        self.reset()

    @property
    def name(self) -> str:
        r""" Get the name of the experiment.

        Returns:
            The name of the experiment.
        """
        return self._name

    @property
    def version(self) -> int:
        r""" Get the experiment version.

        Returns:
            The experiment version if specified else the next version.
        """
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        root_dir = self.root_dir

        try:
            listdir_info = self._fs.listdir(root_dir)
        except OSError:
            logger.warning("Missing logger folder: %s", root_dir)
            return 0

        existing_versions = []
        for listing in listdir_info:
            d = listing["name"]
            bn = os.path.basename(d)
            if self._fs.isdir(d) and bn.startswith("version_"):
                dir_ver = bn.split("_")[1].replace("/", "")
                existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1

    @staticmethod
    def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
        params = _utils_sanitize_params(params)
        # logging of arrays with dimension > 1 is not supported, sanitize as string
        return {k: str(v) if isinstance(v, (torch.Tensor, np.ndarray)) and v.ndim > 1 else v for k, v in params.items()}

    def _sanitize_and_write_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        r""" Just convert to default types and create object ready to be dumped. """
        metrics = dict(
            step=step,
            **{k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
        )
        self.experiment.write(json.dumps(metrics) + "\n")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_experiment"] = None
        return state

    def __setstate__(self, state: Dict[Any, Any]):
        del state["_experiment"]
        self._experiment = None
        self.__dict__.update(state)

    @staticmethod
    def add_logger_specific_args(parser: ArgumentParser):
        r""" Add callback_specific arguments to parser. """
        parser.add_argument(
            '--jsonboard_dir', type=str, required=False, default='jsonboard', help="Where to save logs."
        )
