import os.path

from . import orchestrate as orch

CSV_FILE = "./data/raw/housing-prices-35.csv"
DATA_PATH = "./data/processed/"
TRAIN_PATH = f"{DATA_PATH}train.csv"
VAL_PATH = f"{DATA_PATH}val.csv"
TEST_PATH = f"{DATA_PATH}test.csv"
PROBLEM_TRAIN_PATH = f"{DATA_PATH}p_train.csv"
PROBLEM_VAL_PATH = f"{DATA_PATH}p_val.csv"
PROBLEM_TEST_PATH = f"{DATA_PATH}p_test.csv"
FULL_TRAIN_PATH = f"{DATA_PATH}full_train.csv"
FULL_VAL_PATH = f"{DATA_PATH}full_val.csv"
FULL_TEST_PATH = f"{DATA_PATH}full_test.csv"


def test_createDatasets():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    if not os.path.isfile(FULL_TRAIN_PATH):
        orch.create_processed_dataset(
            category="A",
            train_path=FULL_TRAIN_PATH,
            val_path=FULL_VAL_PATH,
            test_path=FULL_TEST_PATH,
        )
        orch.create_processed_dataset(
            category="H", train_path=TRAIN_PATH, val_path=VAL_PATH, test_path=TEST_PATH
        )
        orch.create_processed_dataset(
            category="C",
            train_path=PROBLEM_TRAIN_PATH,
            val_path=PROBLEM_VAL_PATH,
            test_path=PROBLEM_TEST_PATH,
        )

    assert os.path.isfile(TRAIN_PATH)
    assert os.path.isfile(VAL_PATH)
    assert os.path.isfile(TEST_PATH)
    assert os.path.isfile(PROBLEM_TRAIN_PATH)
    assert os.path.isfile(PROBLEM_VAL_PATH)
    assert os.path.isfile(PROBLEM_TEST_PATH)
    assert os.path.isfile(FULL_TRAIN_PATH)
    assert os.path.isfile(FULL_VAL_PATH)
    assert os.path.isfile(FULL_TEST_PATH)
