from argparse import ArgumentParser


class DefaultConfig:
    r"""
    Keeping all default values together improves readability and editability.
    Do not touch this file unless you want to add something. Possibly subclass this class
    if you want to add some parameters.
    """

    @staticmethod
    def add_defaults_args(parser: ArgumentParser):
        parser.add_argument(
            '--output_dir',
            type=str,
            required=False,
            default="outputs",
            help='Specify a different output folder'
        )
        parser.add_argument(
            '--tensorboard_dir',
            type=str,
            required=False,
            default="tensorboard",
            help="Where tensorboard logs should be saved"
        )
        parser.add_argument(
            '--checkpoints_dir',
            type=str,
            required=False,
            default="checkpoints",
            help="Where checkpoints should be saved"
        )
