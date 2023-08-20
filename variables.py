# Data sources
## raw unzipped archive.zip
CSV_FILE = "./data/raw/housing-prices-35.csv"
DATA_PATH = "./data/processed/"
## data sets for training (only houses but no condos)
TRAIN_PATH = f"{DATA_PATH}train.csv"
VAL_PATH = f"{DATA_PATH}val.csv"
TEST_PATH = f"{DATA_PATH}test.csv"
## data sets for stressing (only condos but no houses)
PROBLEM_TRAIN_PATH = f"{DATA_PATH}p_train.csv"
PROBLEM_VAL_PATH = f"{DATA_PATH}p_val.csv"
PROBLEM_TEST_PATH = f"{DATA_PATH}p_test.csv"
## full data sets
FULL_TRAIN_PATH = f"{DATA_PATH}full_train.csv"
FULL_VAL_PATH = f"{DATA_PATH}full_val.csv"
FULL_TEST_PATH = f"{DATA_PATH}full_test.csv"

# Path to the HTML reports
EVIDENTLY_REPORT_PATH = "./evidently/"

# Feature configuration parameters for the model
CATEGORICAL = []  # ['x_lbt93', 'y_lbt93']
NUMERICAL = ["area_living", "area_land", "n_rooms", "price", "year"]
FEATURES = CATEGORICAL + NUMERICAL
TARGET_FEATURE = ["price"]

# MLFlow configuration parameters
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "housing-prices-experiment"
MLFLOW_MODEL_NAME = "housing-prices-regressor"

# Best parameters for the xgboost model training
BEST_PARAMS = {
    "learning_rate": 0.24672699526038375,  # <0,3
    "max_depth": 21,  # <60
    "min_child_weight": 0.4633648424343051,  # <0,6
    "objective": "reg:squarederror",
    "reg_alpha": 0.07214781548729281,
    "reg_lambda": 0.08286020895000905,  # <0,25
    "seed": 42,
}
