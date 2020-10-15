from transformers import AdamW, get_linear_schedule_with_warmup
from transformers_lightning import utils, models
from pytorch_lightning import _logger as logger


class TransformersModel(models.SuperModel):

    def configure_optimizers(self):
        """
        Instantiate an optimizer on the parameters of self.model.
        A linear scheduler is also instantiated to manage the learning rate.
        """
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
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.learning_rate,
                          eps=self.hparams.adam_epsilon,
                          betas=self.hparams.adam_betas)

        # init scheduler after optional fp16 to get rid of strange warning about optimizer and scheduler steps order
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=self.hparams.max_steps)

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parser):

        parser.add_argument('--do_lower_case', action='store_true')
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--max_sequence_length', type=int, default=128)
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--adam_epsilon', type=float, default=1e-8)
        parser.add_argument('--adam_betas', nargs=2, type=float, default=[0.9, 0.999])
        parser.add_argument('--max_grad_norm', type=float, default=1e-8)
        parser.add_argument('--warmup_steps', type=int, default=10000)

        tmp_args, extra = parser.parse_known_args()
        if tmp_args.learning_rate > 1:
            logger.warning(f"You specified a huge learning rate! Learning rate: {tmp_args.learning_rate}")

        return parser
