import torch
from transformers import BertTokenizer

import pytest
from transformers_lightning.language_modeling.masked_language_modeling import MaskedLanguageModeling, IGNORE_IDX
from transformers_lightning.language_modeling.utils import whole_word_tails_mask

tok = BertTokenizer.from_pretrained('bert-base-cased')
mlm = MaskedLanguageModeling(tok, whole_word_masking=True)


# Test iter dataset work correctly
@pytest.mark.parametrize(
    ["seed", "sentence", "masking"], [
        [0, "how are you man?", [IGNORE_IDX, IGNORE_IDX, 1132, 1128, IGNORE_IDX, IGNORE_IDX, IGNORE_IDX]],
        [
            1, "The quick brown fox jumps over the lazy dog",
            [
                IGNORE_IDX, IGNORE_IDX, IGNORE_IDX, IGNORE_IDX, 17594, IGNORE_IDX, IGNORE_IDX, IGNORE_IDX, IGNORE_IDX,
                IGNORE_IDX, IGNORE_IDX
            ]
        ],
        [
            8, "Be or not to be a superstar",
            [
                IGNORE_IDX, IGNORE_IDX, IGNORE_IDX, IGNORE_IDX, IGNORE_IDX, IGNORE_IDX, IGNORE_IDX, 7688, 10058,
                IGNORE_IDX
            ]
        ], [3, "", [IGNORE_IDX, IGNORE_IDX]],
        [
            4, "share silence or say what you think?",
            [IGNORE_IDX, IGNORE_IDX, 3747, IGNORE_IDX, 1474, 1184, IGNORE_IDX, IGNORE_IDX, IGNORE_IDX, IGNORE_IDX]
        ]
    ]
)
def test_datamodule_cpu(seed, sentence, masking):

    torch.random.manual_seed(seed)
    torch.cuda.random.manual_seed(seed)

    input_ids = torch.tensor([tok.encode(sentence)])
    words_tails_mask = whole_word_tails_mask(input_ids, tok)

    original = input_ids.clone()

    masked, labels = mlm(input_ids, words_tails=words_tails_mask)

    assert torch.all(torch.where(labels != IGNORE_IDX, labels, masked).eq(original))

    labels = labels.tolist()[0]
    assert labels == masking, f"{labels} different from {masking}"
