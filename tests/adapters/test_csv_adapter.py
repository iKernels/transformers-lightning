from argparse import Namespace
from transformers_lightning.adapters import CSVAdapter
import pytest


class ExampleAdapter(CSVAdapter):

    def preprocess_line(self, line: list) -> list:
        return [int(line[0]), int(line[1]), int(line[2]), line[3], line[4], eval(line[5])]


# Test iter dataset work correctly with dp
@pytest.mark.parametrize(
    ["test_file", "line_number", "expected_line", "total_len"], [
        [
            "tests/test_data/test1.tsv", 3,
            [3, 33115, 4156, "How can I reduce tummy fat?", "How do I reduce tummy?", True], 40
        ],
        [
            "tests/test_data/test2.tsv", 54,
            [54, 522296, 3013, "What's it like to take a daily caffeine pill?", "How can I improve digestion?", False],
            96
        ],
        [
            "tests/test_data/test3.tsv", 13,
            [13, 351690, 490931, "How do I withdraw at Quora?", "How can I remove my account from Quora?", True], 17
        ]
    ]
)
def test_datamodule_gpu_dp(test_file, line_number, expected_line, total_len):

    hparams = Namespace()
    adapter = ExampleAdapter(hparams, test_file)
    data = [adapter.preprocess_line(line) for line in iter(adapter)]

    assert data[line_number] == expected_line, (f"Expected: {expected_line}, got: {data[line_number]}")
    assert len(data) == total_len, f"Expected: {total_len}, got: {len(data)}"
