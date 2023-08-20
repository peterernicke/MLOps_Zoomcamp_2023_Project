import os.path

import orchestrate as orch
import variables as v

#CSV_FILE = "./data/raw/housing-prices-35.csv"
#DATA_PATH = "./data/processed/"
#TRAIN_PATH = f"{DATA_PATH}train.csv"
#VAL_PATH = f"{DATA_PATH}val.csv"
#TEST_PATH = f"{DATA_PATH}test.csv"
#PROBLEM_TRAIN_PATH = f"{DATA_PATH}p_train.csv"
#PROBLEM_VAL_PATH = f"{DATA_PATH}p_val.csv"
#PROBLEM_TEST_PATH = f"{DATA_PATH}p_test.csv"
#FULL_TRAIN_PATH = f"{DATA_PATH}full_train.csv"
#FULL_VAL_PATH = f"{DATA_PATH}full_val.csv"
#FULL_TEST_PATH = f"{DATA_PATH}full_test.csv"


def test_createDatasets():
    if not os.path.exists(v.DATA_PATH):
        os.makedirs(v.DATA_PATH)

    if not os.path.isfile(v.FULL_TRAIN_PATH):
        orch.create_processed_dataset(
            category="A",
            train_path=v.FULL_TRAIN_PATH,
            val_path=v.FULL_VAL_PATH,
            test_path=v.FULL_TEST_PATH,
        )
        orch.create_processed_dataset(
            category="H", train_path=v.TRAIN_PATH, val_path=v.VAL_PATH, test_path=v.TEST_PATH
        )
        orch.create_processed_dataset(
            category="C",
            train_path=v.PROBLEM_TRAIN_PATH,
            val_path=v.PROBLEM_VAL_PATH,
            test_path=v.PROBLEM_TEST_PATH,
        )

    assert os.path.isfile(v.TRAIN_PATH)
    assert os.path.isfile(v.VAL_PATH)
    assert os.path.isfile(v.TEST_PATH)
    assert os.path.isfile(v.PROBLEM_TRAIN_PATH)
    assert os.path.isfile(v.PROBLEM_VAL_PATH)
    assert os.path.isfile(v.PROBLEM_TEST_PATH)
    assert os.path.isfile(v.FULL_TRAIN_PATH)
    assert os.path.isfile(v.FULL_VAL_PATH)
    assert os.path.isfile(v.FULL_TEST_PATH)
