""" Keeping all default values together improves editability. """


class DefaultConfig:
    dataset_dir = "input"
    config_dir = "conf"
    cache_dir = "cache"
    output_dir = "outputs"
    results_dir = "results"
    pre_trained_dir = "pre_trained_models"
    tensorboard_dir = "tensorboard"
    checkpoints_dir = "checkpoints"

    @staticmethod
    def add_defaults_args(parser):
        parser.add_argument('--dataset_dir', type=str, required=False, default=DefaultConfig.dataset_dir,
                        help='Specify the dataset files folder')
        parser.add_argument('--config_dir', type=str, required=False, default=DefaultConfig.config_dir,
                            help='Specify a different config folder')
        parser.add_argument('--cache_dir', type=str, required=False, default=DefaultConfig.cache_dir,
                            help='Specify a different cache folder for models and datasets')
        parser.add_argument('--output_dir', type=str, required=False, default=DefaultConfig.output_dir,
                            help='Specify a different output folder')
        parser.add_argument('--pre_trained_dir', type=str, required=False, default=DefaultConfig.pre_trained_dir,
                            help="Default path to save transformer models to")
        parser.add_argument('--tensorboard_dir', type=str, required=False, default=DefaultConfig.tensorboard_dir,
                            help="Where tensorboard logs should be saved")
        parser.add_argument('--checkpoints_dir', type=str, required=False, default=DefaultConfig.checkpoints_dir,
                            help="Where checkpoints should be saved")
        return parser