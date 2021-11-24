import pytest
import torch
from pytorch_lightning import seed_everything
from transformers import BertTokenizer

from transformers_lightning.language_modeling import IGNORE_IDX
from transformers_lightning.language_modeling.swapped_language_modeling import SwappedLanguageModeling

tok = BertTokenizer('tests/data/vocab.txt')
solm = SwappedLanguageModeling(tok, probability=0.4)


@pytest.mark.parametrize(
    ["seed", "sentence", "position_ids", "position_labels"], [
        [
            0, "how are you man?",
            [1, 2, 4, 5, 3, 6, 7],
            [IGNORE_IDX, IGNORE_IDX, 3, 4, 5, IGNORE_IDX, IGNORE_IDX],
        ],
        [
            1, "The quick brown fox jumps over the lazy dog",
            [1, 5, 3, 4, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            [
                IGNORE_IDX, 2, IGNORE_IDX, IGNORE_IDX, 5, IGNORE_IDX, 7, IGNORE_IDX, IGNORE_IDX, IGNORE_IDX,
                IGNORE_IDX, IGNORE_IDX, IGNORE_IDX, 14, IGNORE_IDX, IGNORE_IDX, IGNORE_IDX, IGNORE_IDX, IGNORE_IDX,
                IGNORE_IDX, 21, IGNORE_IDX
            ],
        ],
        [
            8, "Be or not to be a superstar",
            [1, 2, 3, 10, 5, 6, 7, 4, 9, 8, 11],
            [IGNORE_IDX, IGNORE_IDX, IGNORE_IDX, 4, IGNORE_IDX, IGNORE_IDX, IGNORE_IDX, 8, IGNORE_IDX, 10, IGNORE_IDX],
        ],
        [
            3, "",
            [1, 2],
            [IGNORE_IDX, IGNORE_IDX],
        ],
    ]
)
def test_language_model(seed, sentence, position_ids, position_labels):

    seed_everything(seed)

    input_ids = torch.tensor([tok.encode(sentence)])
    new_position_ids, new_position_labels = solm(input_ids)

    position_ids = torch.tensor(position_ids, dtype=torch.long)
    position_labels = torch.tensor(position_labels, dtype=torch.long)

    assert torch.all(torch.eq(new_position_ids, position_ids))
    assert torch.all(torch.eq(new_position_labels, position_labels))
