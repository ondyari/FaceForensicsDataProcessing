import json
from pathlib import Path

__current_directory = Path(__file__).resolve().parent


def flatten(l: list):
    return {item for sublist in l for item in sublist}


TRAIN, TRAIN_NAME = (
    flatten(json.load((__current_directory / "train_dfdc.json").open())),
    "train",
)
VAL, VAL_NAME = flatten(json.load((__current_directory / "val.json").open())), "val"
TEST, TEST_NAME = flatten(json.load((__current_directory / "test.json").open())), "test"
TEST_AIF, TEST_NAME_AIF = (
    flatten(json.load((__current_directory / "test_aif.json").open())),
    "test",
)
