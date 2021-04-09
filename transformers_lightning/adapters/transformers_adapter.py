from argparse import Namespace
from typing import List, Union

import transformers

from transformers_lightning.adapters.csv_adapter import CSVAdapter


class TransformersAdapter(CSVAdapter):
    r"""
    This Adapter adds tokenizing capabilities. The tokenizer of the `transformers` library can be used to
    transformers sentence in sequences of ids. It inherits from CSV because most of the NLP datasets
    are given in this format.

    An example of `preprocess_line` for NLP applications with the `transfomers` library is given below.

    Example:
        >>> def preprocess_line(self, line: list) -> list:
        >>>     results = self.tokenizer.encode_plus(
        >>>         line[1],
        >>>         padding='max_length',
        >>>         max_length=self.hparams.max_sequence_length,
        >>>         truncation=True
        >>>     )
        >>>     results['words_tails'] = self._convert_ids_to_word_tails(results['input_ids'])
        >>>     return results

    Moreover, some useful tools are given as additional methods:
    - `_convert_ids_to_word_tails`: convert a sequence of ids in a list of boolean values where `True`
        represents tail tokens (tokens that starts with '##') and `False` represents first tokens in words.
    """

    def __init__(
        self,
        hparams: Namespace,
        filepath: str,
        tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast] = None,
        **kwargs
    ):
        super().__init__(hparams, filepath, **kwargs)
        self.tokenizer = tokenizer

    def _convert_ids_to_word_tails(self, ids: List[int]):
        r"""
        Convert ids to a mask with True values representing tail words (e.g. starting with '##' in BERT)

        Example:
        >>> sentence = "This is a syntactically correct sentence"
        >>> corresponding_ids = [101, 1188, 1110, 170, 188, 5730, 1777, 11143, 2716, 5663, 5650, 102]
        >>> tokens = ['[CLS]', 'This', 'is', 'a', 's', '##yn', '##ta', '##ctic', '##ally', 'correct', 'sentence', '[SEP]']
        >>> words_tails = [False, False, False, False, False, True, True, True, True, False, False, False]
        """
        return [tok.startswith('##') for tok in self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)]
