from argparse import ArgumentParser, Namespace

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.distributed import rank_zero_warn
from transformers import AdamW

from transformers_lightning import utils
from transformers_lightning.schedulers.linear_scheduler_with_warmup import LinearSchedulerWithWarmup


class TransformersModel(LightningModule):
    r"""
    `TransformersModel` add a ready-to-be-used optimizer function and adds some parameters to
    the command line parser for usual training hyperparameters.
    """

    model: torch.nn.Module
    hparams: Namespace

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

    def forward(self, *args, **kwargs) -> dict:
        r"""
        Simply call the `model` attribute with the given args and kwargs
        """
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        r"""
        Instantiate an optimizer on the parameters of self.model.
        A linear scheduler is also instantiated to manage the learning rate.
        """

        # fix max number of steps
        max_steps = utils.compute_max_steps(self.hparams, self.trainer)

        # get all parameters with names
        all_named_parameters = self.model.named_parameters()

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in all_named_parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in all_named_parameters if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]

        # Define adam optimizer
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
            betas=self.hparams.adam_betas
        )

        # init scheduler after optional fp16 to get rid of strange warning about optimizer and scheduler steps order
        scheduler = LinearSchedulerWithWarmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=max_steps,
            beg_step=self.hparams.beg_scheduler_step
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler':
                {
                    'scheduler': scheduler,    # The LR schduler
                    'interval': 'step',    # The unit of the scheduler's step size
                    'frequency': 1,    # The frequency of the scheduler
                }
        }

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser):
        r"""
        Usual parameters used by AdamW and LinearScheduler. Moreover, it checks the learning rate is at
        reasonable values.
        """
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--max_sequence_length', type=int, default=128)
        parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--adam_epsilon', type=float, default=1e-8)
        parser.add_argument('--adam_betas', nargs=2, type=float, default=[0.9, 0.999])
        parser.add_argument('--max_grad_norm', type=float, default=1e-8)
        parser.add_argument('--warmup_steps', type=int, default=0)
        parser.add_argument('--beg_scheduler_step', type=int, default=0)

        tmp_args, _ = parser.parse_known_args()
        if tmp_args.learning_rate > 1:
            rank_zero_warn(f"You specified a huge learning rate! Learning rate: {tmp_args.learning_rate}")

        return parser
