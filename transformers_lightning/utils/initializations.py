import os

def init_folders(hparams):
    # create output dir if it does not exist
    if not os.path.isdir(hparams.output_dir):
        os.mkdir(hparams.output_dir)
