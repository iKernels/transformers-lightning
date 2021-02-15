import torch
from transformers import BertTokenizer

import pytest
from transformers_lightning.language_modeling import IGNORE_IDX
from transformers_lightning.language_modeling.random_token_substitution import RandomTokenSubstitution
from transformers_lightning.language_modeling.utils import whole_word_tails_mask

tok = BertTokenizer.from_pretrained('bert-base-cased')
rts = RandomTokenSubstitution(tok, whole_word_swapping=True)


# Test iter dataset work correctly
@pytest.mark.parametrize(
    ["seed", "sentence", "masking", "new_ids"], [
        [0, "how are you man?", [IGNORE_IDX, 0, 1, 1, 0, 0, IGNORE_IDX], [101, 1293, 28135, 17785, 1299, 136, 102]],
        [
            1, "The quick brown fox jumps over the lazy dog", [IGNORE_IDX, 0, 0, 0, 1, 0, 0, 0, 0, 0, IGNORE_IDX],
            [101, 1109, 3613, 3058, 18156, 15457, 1166, 1103, 16688, 3676, 102]
        ],
        [
            8, "Be or not to be a superstar", [IGNORE_IDX, 0, 0, 0, 0, 0, 0, 1, 1, IGNORE_IDX],
            [101, 4108, 1137, 1136, 1106, 1129, 170, 4667, 20540, 102]
        ], [3, "", [IGNORE_IDX, IGNORE_IDX], [101, 102]],
        [
            4, "share silence or say what you think?", [IGNORE_IDX, 0, 1, 0, 1, 1, 0, 0, 0, IGNORE_IDX],
            [101, 2934, 13287, 1137, 24438, 26891, 1128, 1341, 136, 102]
        ]
    ]
)
def test_datamodule_cpu(seed, sentence, masking, new_ids):

    torch.random.manual_seed(seed)
    torch.cuda.random.manual_seed(seed)

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
