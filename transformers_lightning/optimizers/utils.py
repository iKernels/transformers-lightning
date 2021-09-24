from typing import Generator


def get_parameters_grouped_for_weight_decay(named_parameters: Generator, weight_decay: float = 0.0):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in named_parameters if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in named_parameters if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    return optimizer_grouped_parameters


def named_parameters_to_parameters(named_parameters: Generator) -> Generator:
    r""" Extract params only from named parameters. """
    for _, params in named_parameters:
        yield params
