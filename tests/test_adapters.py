from argparse import Namespace

import pytest

from tests.helpers import DummyCSVAdapter


@pytest.mark.parametrize(
    "test_file, line_number, expected_line, total_len", (
        ["tests/data/file-1.tsv", 3, [3, 33115, 4156, "This is a question 3", "This is an answer 3", True], 17],
        ["tests/data/file-2.tsv", 54, [54, 522296, 3013, "This is a question 54", "This is an answer 54", False], 96],
        ["tests/data/file-3.tsv", 13, [13, 351690, 490931, "This is a question 13", "This is an answer 13", True], 40]
    )
)
def test_csv_adapter(test_file, line_number, expected_line, total_len):

    hyperparameters = Namespace(output_dir='/tmp/output')
    adapter = DummyCSVAdapter(hyperparameters, test_file)
    data = [adapter.preprocess_line(line) for line in iter(adapter)]

    assert data[line_number] == expected_line, (f"Expected: {expected_line}, got: {data[line_number]}")
    assert len(data) == total_len, f"Expected: {total_len}, got: {len(data)}"
