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

EVIDENTLY_REPORT_PATH = "./evidently/"

CATEGORICAL = []  # ['x_lbt93', 'y_lbt93']
NUMERICAL = ["area_living", "area_land", "n_rooms", "price", "year"]
FEATURES = CATEGORICAL + NUMERICAL
TARGET_FEATURE = ["price"]

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "housing-prices-experiment"
MLFLOW_MODEL_NAME = "housing-prices-regressor"

BEST_PARAMS = {
    "learning_rate": 0.24672699526038375,  # <0,3
    "max_depth": 21,  # <60
    "min_child_weight": 0.4633648424343051,  # <0,6
    "objective": "reg:squarederror",
    "reg_alpha": 0.07214781548729281,
    "reg_lambda": 0.08286020895000905,  # <0,25
    "seed": 42,
}
