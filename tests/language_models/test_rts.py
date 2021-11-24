import pytest
import torch
from pytorch_lightning import seed_everything
from transformers import BertTokenizer

from transformers_lightning.language_modeling import IGNORE_IDX
from transformers_lightning.language_modeling.random_token_substitution import RandomTokenSubstitution
from transformers_lightning.language_modeling.utils import whole_word_tails_mask

tok = BertTokenizer('tests/data/vocab.txt')
rts = RandomTokenSubstitution(tok, whole_word_swapping=True)


@pytest.mark.parametrize(
    ["seed", "sentence", "masking", "new_ids"], [
        [0, "how are you man?", [IGNORE_IDX, 0, 1, 1, 0, 0, IGNORE_IDX], [2, 1135, 1343, 961, 1164, 35, 3]],
        [
            1, "The quick brown fox jumps over the lazy dog", [IGNORE_IDX] + [0] * 20 + [IGNORE_IDX],
            [
                2, 1002, 59, 1232, 1600, 1249, 1835, 48, 1086, 1601, 52, 1825,
                1367, 1021, 1064, 1002, 1480, 1486, 1106, 1085, 1296, 3
            ]
        ],
        [
            8, "Be or not to be a superstar", [IGNORE_IDX, 0, 0, 0, 0, 0, 0, 1, 1, 1, IGNORE_IDX],
            [2, 1028, 1036, 1031, 1006, 1028, 43, 1932, 753, 2577, 3]
        ], [3, "", [IGNORE_IDX, IGNORE_IDX], [2, 3]],
        [
            4, "share silence or say what you think?", [IGNORE_IDX, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, IGNORE_IDX],
            [2, 2751, 723, 2625, 2164, 1584, 358, 1036, 1366, 1060, 1023, 1234, 35, 3]
        ]
    ]
)
def test_language_model(seed, sentence, masking, new_ids):

    seed_everything(seed)

    input_ids = torch.tensor([tok.encode(sentence)])
    words_tails_mask = whole_word_tails_mask(input_ids, tok)

    original = input_ids.clone()

    swapped, labels = rts(input_ids, words_tails=words_tails_mask)

    assert torch.all(torch.eq(swapped[labels != 1], original[labels != 1]))
    assert torch.all(torch.ne(swapped[labels == 1], original[labels == 1]))

    labels = labels.tolist()[0]
    swapped = swapped.tolist()[0]

    assert swapped == new_ids, f"{swapped} different from {new_ids}"
    assert labels == masking, f"{labels} different from {masking}"
