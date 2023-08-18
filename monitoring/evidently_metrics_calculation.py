import io
import time
import uuid
import random
import logging
import datetime
import xgboost as xgb
import numpy as np
import scipy
import sklearn
import mlflow

import pytz
import joblib
import pandas as pd
import psycopg
from prefect import flow, task, get_run_logger
from evidently import ColumnMapping
from evidently.report import Report

from mlflow.data.pandas_dataset import PandasDataset
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException

from sklearn.feature_extraction import DictVectorizer

# maybe also interesting (from my side of view)
from evidently.metrics import(
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetSummaryMetric,
    DatasetCorrelationsMetric,
    DatasetMissingValuesMetric
)

EVIDENTLY_REPORT_PATH = "./evidently/"
DATA_PATH           = "../data/processed/"
TRAIN_PATH          = f"{DATA_PATH}train.csv"
VAL_PATH            = f"{DATA_PATH}val.csv"
TEST_PATH           = f"{DATA_PATH}test.csv"
PROBLEM_TRAIN_PATH  = f"{DATA_PATH}p_train.csv"
PROBLEM_VAL_PATH    = f"{DATA_PATH}p_val.csv"
PROBLEM_TEST_PATH   = f"{DATA_PATH}p_test.csv"
FULL_TRAIN_PATH     = f"{DATA_PATH}full_train.csv"
FULL_VAL_PATH       = f"{DATA_PATH}full_val.csv"
FULL_TEST_PATH      = f"{DATA_PATH}full_test.csv"

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "housing-prices-experiment"
MLFLOW_MODEL_NAME = "housing-prices-regressor"

CATEGORICAL = []#['x_lbt93', 'y_lbt93']
NUMERICAL = ["area_living", "area_land", "n_rooms", "price", "year"]
FEATURES = CATEGORICAL + NUMERICAL

def add_features(
    df_train: pd.DataFrame, df_val: pd.DataFrame
) -> tuple(
    [
        scipy.sparse._csr.csr_matrix,
        scipy.sparse._csr.csr_matrix,
        np.ndarray,
        np.ndarray,
        sklearn.feature_extraction.DictVectorizer,
    ]
):
    features = FEATURES

    dv = DictVectorizer()

    train_dicts = df_train[features].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[features].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train = df_train["price"].values
    y_val = df_val["price"].values
    return X_train, X_val, y_train, y_val, dv

#logging.basicConfig(
#    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
#)

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
    timestamp timestamp,
    prediction_drift float,
    num_drifted_columns varchar,
    share_missing_val float
)
"""

mlflow_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# load reference data and model
ref_data = pd.read_csv(TRAIN_PATH)
model_uri = "./models/models_mlflow"
booster = mlflow.xgboost.load_model(model_uri)

num_features = NUMERICAL
cat_features = CATEGORICAL
target = "price"

column_mapping = ColumnMapping(
    prediction="prediction",
    numerical_features=num_features,
    categorical_features=cat_features,
    target=None,
)

report = Report(
    metrics=[
        ColumnDriftMetric(column_name="prediction"),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        # DatasetCorrelationsMetric, DatasetSummaryMetric
        DatasetCorrelationsMetric(),
        DatasetSummaryMetric(),
    ]
)

@task
def prep_db():
    with psycopg.connect(
        "host=localhost port=5432 user=postgres password=example", autocommit=True
    ) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("create database test;")
        with psycopg.connect(
            "host=localhost port=5432 dbname=test user=postgres password=example"
        ) as conn:
            conn.execute(create_table_statement)

# using curr because we're going to insert values in the specific position of the cursor
# create some report and derive needed values
# the additional parameter i is used to calculate metrics
@task
def calculate_metrics_postgresql(curr, i, problematic_data):
    current_data = problematic_data

    X_train, X_test, y_train, y_test, _ = add_features(ref_data, current_data)
    test = xgb.DMatrix(X_test, label=y_test)
    current_data['prediction'] = booster.predict(test)

    ref = xgb.DMatrix(X_train, label=y_train)
    ref_preds = booster.predict(ref)
    ref_data['prediction'] = ref_preds

    report.run(
        reference_data=ref_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    result = report.as_dict()

    # deriving some values (prediction drift)
    prediction_drift = result["metrics"][0]["result"]["drift_score"]
    # number of drifted columns
    num_drifted_columns = result["metrics"][1]["result"]["number_of_drifted_columns"]
    # share of missing values
    share_missing_val = result["metrics"][2]["result"]["current"][
        "share_of_missing_values"
    ]

    curr.execute(
        "insert into dummy_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_val) values (%s, %s, %s, %s)",
        (
            datetime.datetime.now() + datetime.timedelta(hours=i),
            prediction_drift,
            num_drifted_columns,
            share_missing_val,
        ),
    )

@flow(log_prints=True)
def batch_monitoring_backfill():
    prep_db()
    logger = get_run_logger()
    position = ["First", "Second", "Third"]
    
    # simulate production usage of our batch service while using different data sets
    logger.info("Start simulating production usage...")   
    with psycopg.connect(
        "host=localhost port=5432 dbname=test user=postgres password=example",
        autocommit=True,
    ) as conn:
        delta = 0
        #logger.info("Processing data from {} data set...".format(PROBLEM_VAL_PATH))
        logger.info("Processing data from %s..." % PROBLEM_VAL_PATH)
        problematic_data = pd.read_csv(PROBLEM_VAL_PATH)
        for i in range(0, 3):
            with conn.cursor() as curr:
                calculate_metrics_postgresql(curr, delta, problematic_data)
                delta += 10
            logger.info(f"{position[i]} run processed")
            #print(f"data from {PROBLEM_VAL_PATH} sent")

        logger.info("Processing data from %s ..." % VAL_PATH)
        problematic_data = pd.read_csv(VAL_PATH)
        for i in range(0, 3):
            with conn.cursor() as curr:
                calculate_metrics_postgresql(curr, delta, problematic_data)
                delta += 10
            logger.info(f"{position[i]} run processed")
            #print(f"data from {VAL_PATH} sent")
        
        logger.info("Processing data from %s ..." % TRAIN_PATH)
        problematic_data = pd.read_csv(TRAIN_PATH)
        for i in range(0, 3):
            with conn.cursor() as curr:
                calculate_metrics_postgresql(curr, delta, problematic_data)
                delta += 10
            logger.info(f"{position[i]} run processed")
            #print(f"data from {TRAIN_PATH} sent")

if __name__ == "__main__":
    batch_monitoring_backfill()
