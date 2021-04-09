from argparse import ArgumentParser


class DefaultConfig:
    """
    Keeping all default values together improves readability and editability. 
    Do not touch this tfile unless you want to add something. Possibly subclass this class
    if you want to add some foundamental parameter.
    """

    output_dir: str = "outputs"
    pre_trained_dir: str = "pre_trained_models"
    tensorboard_dir: str = "tensorboard"
    checkpoints_dir: str = "checkpoints"

    @staticmethod
    def add_defaults_args(parser: ArgumentParser):
        parser.add_argument(
            '--output_dir',
            type=str,
            required=False,
            default=DefaultConfig.output_dir,
            help='Specify a different output folder'
        )
        parser.add_argument(
            '--pre_trained_dir',
            type=str,
            required=False,
            default=DefaultConfig.pre_trained_dir,
            help="Default path to save transformer models to"
        )
        parser.add_argument(
            '--tensorboard_dir',
            type=str,
            required=False,
            default=DefaultConfig.tensorboard_dir,
            help="Where tensorboard logs should be saved"
        )
        parser.add_argument(
            '--checkpoints_dir',
            type=str,
            required=False,
            default=DefaultConfig.checkpoints_dir,
            help="Where checkpoints should be saved"
        )
        return parser
