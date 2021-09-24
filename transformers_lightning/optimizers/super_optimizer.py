from argparse import ArgumentParser, Namespace


class SuperOptimizer:
    r""" High level interface for optimizers to be used with transformers.
    Adds methods to define hyperparameters from the command line. """

    def __init__(self, hyperparameters: Namespace, *args, **kwargs) -> None:
        r""" First argument should always be the hyperparameters namespace, other arguments are
        optimizer-specific. """
        super().__init__(*args, **kwargs)    # forward init to constructor of real optimizer
        self.hyperparameters = hyperparameters

    @staticmethod
    def add_optimizer_specific_args(parser: ArgumentParser):
        r""" Add here the hyperparameters used by your optimizer. """
