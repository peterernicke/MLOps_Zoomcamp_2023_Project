import pickle
import os.path
import pathlib

from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

import numpy as np
import scipy
import mlflow
import pandas as pd
import seaborn as sns
import sklearn
import xgboost as xgb
import matplotlib.pyplot as plt
# conda install -c conda-forge prefect
# prefect server start
from prefect import flow, task
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.feature_extraction import DictVectorizer

CSV_FILE = "./data/raw/housing-prices-35.csv"
# print(CSV_FILE)

TRAIN_PATH = "./data/processed/train.csv"
VAL_PATH = "./data/processed/val.csv"
TEST_PATH = "./data/processed/test.csv"

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "housing-prices-experiment"


@task
def create_datasets(csv_file: str = CSV_FILE):
    df = pd.read_csv(csv_file)

    # remove all lines with category equals "C" for condo, because we only want to predict house prices (category equals "H")
    # that reduce the data to 99169 lines
    # df[df.category == "C"].count()
    df.drop(df.loc[df["category"] == "C"].index, inplace=True)
    # actually I wanted also exclude every data set where there is an area_land > 0
    # but that results in too less data sets, so I have to leave it as it is...
    # df[df.area_land == 0].count()

    # drop duplicate data sets --> reduces the whole data set to 57073
    df = df.drop_duplicates()

    # little bit of feature engineering
    df["year"] = df["date"]
    df.year = df.year.apply(lambda td: td[:4])

    num_train = int((df.date.count()) * 0.7)
    train = df.iloc[:num_train]

    num_val = int((df.date.count()) * 0.2)
    val = df.iloc[num_train + 1 : num_train + 1 + num_val]

    test = df.iloc[num_train + 1 + num_val + 1 :]

    if not os.path.isfile(TRAIN_PATH):
        train.to_csv(path_or_buf=TRAIN_PATH)

    if not os.path.isfile(VAL_PATH):
        val.to_csv(path_or_buf=VAL_PATH)

    if not os.path.isfile(TEST_PATH):
        test.to_csv(path_or_buf=TEST_PATH)


@task(retries=3, retry_delay_seconds=2)
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
    # categorical = ['x_lbt93', 'y_lbt93']
    numerical = ["area_living", "area_land", "n_rooms", "price", "year"]
    # features = categorical + numerical
    features = numerical

    # df_train[categorical] = df_train[categorical].astype(str)
    # df_val[categorical] = df_val[categorical].astype(str)

    dv = DictVectorizer()

    train_dicts = df_train[features].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[features].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train = df_train["price"].values
    y_val = df_val["price"].values
    return X_train, X_val, y_train, y_val, dv


@task(log_prints=True)
def train_best_model(
    X_train: scipy.sparse._csr.csr_matrix,
    X_val: scipy.sparse._csr.csr_matrix,
    y_train: np.ndarray,
    y_val: np.ndarray,
    dv: sklearn.feature_extraction.DictVectorizer,
) -> None:
    """train a model with best hyperparams and write everything out"""

    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            # <0,3
            "learning_rate": 0.24672699526038375,
            # <60
            "max_depth": 21,
            # <0,6
            "min_child_weight": 0.4633648424343051,
            "objective": "reg:squarederror",
            "reg_alpha": 0.07214781548729281,
            # <0,25
            "reg_lambda": 0.08286020895000905,
            "seed": 42,
        }

        mlflow.set_tag("model", "xgboost")

        mlflow.log_param("train-data-path", TRAIN_PATH)
        mlflow.log_param("valid-data-path", VAL_PATH)
        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, "validation")],
            early_stopping_rounds=20,
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
        #prec = precision_score(y_val, y_pred)
        #mlflow.log_metric("precision", prec)
        #recall = recall_score(y_val, y_pred)
        #mlflow.log_metric("recall", recall)
        #f1 = f1_score(y_val, y_pred)
        #mlflow.log_metric("f1", f1)
        #acc = accuracy_score(y_val, y_pred)
        #mlflow.log_metric("accuracy", acc)
        r2 = r2_score(y_val, y_pred)
        mlflow.log_metric("r2score", r2)

        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
    return None

@task
def promote_model():
    
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    print(client.search_experiments())


@flow
def main_flow(
    # train_path: str = "./data/green_tripdata_2021-01.parquet",
    # val_path: str = "./data/green_tripdata_2021-02.parquet",
) -> None:
    """The main training pipeline"""

    create_datasets()

    """Read data into DataFrame"""
    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)

    # MLflow settings
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load
    #df_train = read_data(train_path)
    #df_val = read_data(val_path)

    # Transform
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)

    # Train
    train_best_model(X_train, X_val, y_train, y_val, dv)

    # promote the model
    promote_model()


if __name__ == "__main__":
    main_flow()
