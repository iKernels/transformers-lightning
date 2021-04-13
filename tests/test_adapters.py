from argparse import Namespace
import pytest

from transformers import AutoTokenizer

from tests.helpers import DummyCSVAdapter, DummyTransformersAdapter


# Test iter dataset work correctly with dp
@pytest.mark.parametrize(
    "test_file, line_number, expected_line, total_len", (
        [
            "tests/data/test1.tsv", 3,
            [3, 33115, 4156, "How can I reduce tummy fat?", "How do I reduce tummy?", True], 40
        ],
        [
            "tests/data/test2.tsv", 54,
            [54, 522296, 3013, "What's it like to take a daily caffeine pill?", "How can I improve digestion?", False],
            96
        ],
        [
            "tests/data/test3.tsv", 13,
            [13, 351690, 490931, "How do I withdraw at Quora?", "How can I remove my account from Quora?", True], 17
        ]
    )
)
def test_csv_adapter(test_file, line_number, expected_line, total_len):

    hparams = Namespace(output_dir='/tmp/output')
    adapter = DummyCSVAdapter(hparams, test_file)
    data = [adapter.preprocess_line(line) for line in iter(adapter)]

    assert data[line_number] == expected_line, (f"Expected: {expected_line}, got: {data[line_number]}")
    assert len(data) == total_len, f"Expected: {total_len}, got: {len(data)}"



# Test iter dataset work correctly with dp
@pytest.mark.parametrize(
    "test_file, line_number, expected_ids, expected_word_tails", (
        [
            "tests/data/test1.tsv", 3,
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
            "tests/data/test2.tsv", 54,
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
            "tests/data/test3.tsv", 13,
            [
                101, 1731, 1202, 146, 10570, 1120, 154, 11848, 1611, 136, 102, 1731, 1169, 146, 5782, 1139, 3300, 1121,
                154, 11848, 1611, 136, 102
            ],
            [
                False, False, False, False, False, False, False, True, True, False, False, False, False, False, False,
                False, False, False, False, True, True, False, False
            ]
        ]
    )
)
def test_transformers_adapter(test_file, line_number, expected_ids, expected_word_tails):

    hparams = Namespace(output_dir='/tmp/output', padding='do_not_pad')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    adapter = DummyTransformersAdapter(hparams, test_file, tokenizer=tokenizer)
    data = [adapter.preprocess_line(line) for line in iter(adapter)]

    assert data[line_number]['input_ids'] == expected_ids, (
        f"Expected: {expected_ids}, got: {data[line_number]['input_ids']}"
    )
    assert data[line_number]['words_tails'] == expected_word_tails, (
        f"Expected: {expected_word_tails}, got: {data[line_number]['words_tails']}"
    )
