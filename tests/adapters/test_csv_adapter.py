import csv

from argparse import Namespace
from transformers_lightning.adapters import CSVAdapter
from transformers_lightning.utils import strip_lines
import pytest


class ExampleAdapter(CSVAdapter):

    def __iter__(self):
        with open(self.filepath, "r") as fi:
            # use utils.strip_lines to emulate skip_blank_lines of pd.DataFrame
            reader = csv.reader(
                strip_lines(fi),
                delimiter=self.delimiter,
                quoting=self.quoting,
                quotechar=self.quotechar
            )
            for line in reader:
                yield [int(line[0]), int(line[1]), int(line[2]), line[3], line[4], eval(line[5])]
        

    def preprocess_line(self, line: list) -> list:
        return line

# Test iter dataset work correctly with dp
@pytest.mark.parametrize(
    ["test_file", "line_number", "expected_line", "total_len"], [
        ["test1.tsv", 3, [3, 33115, 4156, "How can I reduce tummy fat?", "How do I reduce tummy?", True], 40],
        ["test2.tsv", 54, [54, 522296, 3013, "What's it like to take a daily caffeine pill?", "How can I improve digestion?", False], 96],
        ["test3.tsv", 13, [13, 351690, 490931, "How do I withdraw at Quora?", "How can I remove my account from Quora?", True], 17]
    ]
)
def test_datamodule_gpu_dp(test_file, line_number, expected_line, total_len):

    hparams = Namespace(
        dataset_dir="tests/test_data"
    )
    adapter = ExampleAdapter(hparams, test_file, delimiter="\t")
    data = list(iter(adapter))

    assert data[line_number] == expected_line, (
        f"Expected: {expected_line}, got: {data[line_number]}"
    )
    assert len(data) == total_len, f"Expected: {total_len}, got: {len(data)}"


