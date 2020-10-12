"""
Build config, models and tokenizers of the transformers library
"""

def get_config(config_class, hparams, pre_trained_name=None, **kwargs):
    """
    Load config from scratch or from a pre-trained version.
    """
    if pre_trained_name is not None:
        kwargs['cache_dir'] = hparams.cache_dir
        return config_class.from_pretrained(pre_trained_name, **kwargs)
    return config_class(**kwargs)

def get_model(model_class, hparams, pre_trained_name=None, **kwargs):
    """
    Load model from scratch or from a checkpoint.
    """
    if pre_trained_name is not None:
        assert 'cache_dir' in hparams, \
            "cache_dir is required in params when starting from a pre-trained ckpt"
        kwargs['cache_dir'] = hparams.cache_dir
        return model_class.from_pretrained(pre_trained_name, **kwargs)
    return model_class(**kwargs)

def get_tokenizer(tokenizer_class, hparams, pre_trained_name=None, **kwargs):
    """
    Load tokenizer from scratch or from a checkpoint.
    """
    kwargs['do_lower_case'] = hparams.do_lower_case
    if pre_trained_name is not None:
        kwargs['cache_dir'] = hparams.cache_dir
        return tokenizer_class.from_pretrained(pre_trained_name, **kwargs)
    return tokenizer_class(**kwargs)
