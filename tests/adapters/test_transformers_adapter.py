from argparse import Namespace

import transformers
from transformers_lightning.adapters import TransformersAdapter
import pytest


class ExampleAdapter(TransformersAdapter):

    def preprocess_line(self, line: list) -> list:
        line = [int(line[0]), int(line[1]), int(line[2]), line[3], line[4], eval(line[5])]

        results = self.tokenizer.encode_plus(line[3], line[4], truncation=True)
        results['words_tails'] = self._convert_ids_to_word_tails(results['input_ids'])
        return results


# Test iter dataset work correctly with dp
@pytest.mark.parametrize(
    ["test_file", "line_number", "expected_ids", "expected_word_tails"], [
        [
            "tests/test_data/test1.tsv", 3,
            [
                101, 1731, 1169, 146, 4851, 189, 1818, 4527, 7930, 136, 102, 1731, 1202, 146, 4851, 189, 1818, 4527,
                136, 102
            ],
            [
                False, False, False, False, False, False, True, True, False, False, False, False, False, False, False,
                False, True, True, False, False
            ]
        ],
        [
            "tests/test_data/test2.tsv", 54,
            [
                101, 1327, 112, 188, 1122, 1176, 1106, 1321, 170, 3828, 11019, 15475, 2042, 21822, 136, 102, 1731, 1169,
                146, 4607, 11902, 2556, 1988, 136, 102
            ],
            [
                False, False, False, False, False, False, False, False, False, False, False, True, True, False, False,
                False, False, False, False, False, False, True, True, False, False
            ]
        ],
        [
            "tests/test_data/test3.tsv", 13,
            [
                101, 1731, 1202, 146, 10570, 1120, 154, 11848, 1611, 136, 102, 1731, 1169, 146, 5782, 1139, 3300, 1121,
                154, 11848, 1611, 136, 102
            ],
            [
                False, False, False, False, False, False, False, True, True, False, False, False, False, False, False,
                False, False, False, False, True, True, False, False
            ]
        ]
    ]
)
def test_datamodule_gpu_dp(test_file, line_number, expected_ids, expected_word_tails):

    hparams = Namespace()
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-cased')
    adapter = ExampleAdapter(hparams, test_file, tokenizer=tokenizer)
    data = [adapter.preprocess_line(line) for line in iter(adapter)]

    assert data[line_number]['input_ids'] == expected_ids, (
        f"Expected: {expected_ids}, got: {data[line_number]['input_ids']}"
    )
    assert data[line_number]['words_tails'] == expected_word_tails, (
        f"Expected: {expected_word_tails}, got: {data[line_number]['words_tails']}"
    )
