import requests
import random

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric

from joblib import load, dump
from tqdm import tqdm

import pickle
import os.path
import pathlib

from mlflow.data.pandas_dataset import PandasDataset
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from datetime import date, datetime

import numpy as np
import scipy
import mlflow
import pandas as pd
import seaborn as sns
import sklearn
import xgboost as xgb

from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.feature_extraction import DictVectorizer

CSV_FILE = "./data/raw/housing-prices-35.csv"

EVIDENTLY_REPORT_PATH = "./evidently/"
DATA_PATH           = "./data/processed/"
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

BEST_PARAMS = {
            "learning_rate": 0.24672699526038375,           # <0,3
            "max_depth": 21,                                # <60
            "min_child_weight": 0.4633648424343051,         # <0,6
            "objective": "reg:squarederror",
            "reg_alpha": 0.07214781548729281,
            "reg_lambda": 0.08286020895000905,              # <0,25
            "seed": 42,
            }

@task
def setup():
    # Create data folder and evidently report folder
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    if not os.path.exists(EVIDENTLY_REPORT_PATH):
        os.makedirs(EVIDENTLY_REPORT_PATH)

@task
def create_processed_dataset(
    source_csv_file: str = CSV_FILE,
    category: str = "",                 # H-houses, C-Condo, A-All
    train_path: str = "",
    val_path: str = "",
    test_path: str = "",
):
    df = pd.read_csv(source_csv_file)
    if category == "C":
        df.drop(df.loc[df["category"] == "H"].index, inplace=True)
    elif category == "H":
        df.drop(df.loc[df["category"] == "C"].index, inplace=True)
    
    df = df.drop_duplicates()
    df["year"] = df["date"]
    df.year = df.year.apply(lambda td: td[:4])

    num_train = int((df.date.count()) * 0.7)
    train = df.iloc[:num_train]

    num_val = int((df.date.count()) * 0.2)
    val = df.iloc[num_train + 1 : num_train + 1 + num_val]

    test = df.iloc[num_train + 1 + num_val + 1 :]       

    if not os.path.isfile(train_path):
        train.to_csv(path_or_buf=train_path)

    if not os.path.isfile(val_path):
        val.to_csv(path_or_buf=val_path)

    if not os.path.isfile(test_path):
        test.to_csv(path_or_buf=test_path)

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

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)        

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
    features = FEATURES

    # df_train[CATEGORICAL] = df_train[CATEGORICAL].astype(str)
    # df_val[CATEGORICAL] = df_val[CATEGORICAL].astype(str)

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
    report_type = "Train"
    df = pd.read_csv(TRAIN_PATH)
    dataset: PandasDataset = mlflow.data.from_pandas(df, source=TRAIN_PATH)

    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = BEST_PARAMS

        mlflow.set_tag("model", "xgboost")
        # Experimental: This function may change or be removed in a future release without warning.
        mlflow.log_input(dataset, context="training")
        
        mlflow.log_param("train-data-path", TRAIN_PATH)
        mlflow.log_param("valid-data-path", VAL_PATH)
        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            #num_boost_round=100,
            num_boost_round=10,
            evals=[(valid, "validation")],
            #early_stopping_rounds=20,
            early_stopping_rounds=10,
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
        r2 = r2_score(y_val, y_pred)
        mlflow.log_metric("r2score", r2)

        monitor_model(report_type, pd.read_csv(TRAIN_PATH), pd.read_csv(VAL_PATH), booster, train, valid)

        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        markdown__rmse_report = f"""# RMSE Report

        ## Summary

        Housing Prices France Prediction

        ## RMSE XGBoost Model

        |   Date    |  RMSE  |
        |:----------|-------:|
        | {date.today()} | {rmse:.2f} |
        """

        create_markdown_artifact(
            key="housing-prices-report", markdown=markdown__rmse_report
        )

    return None

@task
def register_model(mlflow_client):  
    
    run = mlflow_client.search_runs(
        experiment_ids='1',
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
    )
    run_id = run[0].info.run_id
    #print(run_id)
    model_uri = f"runs:/{run_id}/models_mlflow"
    #model_uri = f"runs:/{run_id}/model"

    registered_models = mlflow.search_registered_models()

    if not registered_models:
        print("register model")
        mlflow.register_model(model_uri=model_uri, name=MLFLOW_MODEL_NAME)
    else:
        try:
            for model in registered_models:
                if model.name == MLFLOW_MODEL_NAME:
                    lv = model.latest_versions
                    for model_version in lv:
                        if model_version.run_id == run_id:
                            print("model is already versioned")
                            raise StopIteration()
                    print("register model")
                    mlflow.register_model(model_uri=model_uri, name=MLFLOW_MODEL_NAME)
        except StopIteration:
            pass

def get_best_run(mlflow_client):
    #latest_versions = mlflow_client.get_latest_versions(name=MLFLOW_MODEL_NAME)

    # get the registered run with the best rsme
    registered_models = mlflow_client.search_registered_models()
    best_run_id = ""
    best_rsme = 100000
    best_stage = ""

    for m in registered_models:
        lv = m.latest_versions
        
        for version in lv:
            if version.current_stage == "Production":
                best_run_id = version.run_id
                run = mlflow_client.get_run(version.run_id)
                best_rsme = int(run.data.metrics['rmse'])
                best_stage = "Production"
                break
            elif version.current_stage == "Staging":
                if ((best_stage == "") | (best_stage == "None")):
                    best_run_id = version.run_id
                    run = mlflow_client.get_run(version.run_id)
                    best_rsme = int(run.data.metrics['rmse'])
                    best_stage = "Staging"
                else:
                    tmp_run_id = version.run_id
                    run = mlflow_client.get_run(version.run_id)
                    tmp_rsme = int(run.data.metrics['rmse'])
                    if tmp_rsme < best_rsme:
                        best_rsme = tmp_rsme
                        best_stage = "Staging"
                        best_run_id = tmp_run_id
            elif version.current_stage == "None":
                if best_stage == "":
                    best_run_id = version.run_id
                    run = mlflow_client.get_run(version.run_id)
                    best_rsme = int(run.data.metrics['rmse'])
                    best_stage = "None"
                else:
                    tmp_run_id = version.run_id
                    run = mlflow_client.get_run(version.run_id)
                    tmp_rsme = int(run.data.metrics['rmse'])
                    if tmp_rsme < best_rsme:
                        best_rsme = tmp_rsme
                        best_stage = "None"
                        best_run_id = tmp_run_id

    return best_run_id, best_rsme, best_stage

@task
def promote_model(mlflow_client):
    # this function just transitions the first model to Staging
    latest_versions = mlflow_client.get_latest_versions(name=MLFLOW_MODEL_NAME)

    # get the registered run with the best rsme
    best_run_id, best_rsme, _ = get_best_run(mlflow_client)

    #for version in latest_versions:
    #    print(f"version: {version.version}, stage: {version.current_stage}")
    if latest_versions:

        for version in latest_versions:
            run = mlflow_client.get_run(version.run_id)
            rsme = int(run.data.metrics['rmse'])
            new_stage = "Staging"

            if ((rsme < best_rsme) & (version.current_stage == "None")):
                # transition model to next stage
                transition_model(mlflow_client, version.version, new_stage, False)
            elif ((version.run_id == best_run_id)& (version.current_stage == "None")):
                # transition model to next stage
                transition_model(mlflow_client, version.version, new_stage, False)

def monitor_model(report_type, train_data, val_data, model, train, valid):
    today = datetime.now()
    today = f"{today.year}-{today.month:02d}-{today.day:02d}-{today.hour:02d}:{today.minute:02d}"
    report_name = f"Evidently{report_type}Report-{today}.html"
    report_path = f"{EVIDENTLY_REPORT_PATH}{report_name}"

    train_preds = model.predict(train)
    train_data['prediction'] = train_preds

    val_preds = model.predict(valid)
    val_data['prediction'] = val_preds

    column_mapping = ColumnMapping(
        target=None,
        prediction='prediction',
        numerical_features=NUMERICAL,
        categorical_features=CATEGORICAL
    )

    report = Report(metrics=[
        ColumnDriftMetric(column_name='price'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric()
    ]
    )

    report.run(reference_data=train_data, current_data=val_data, column_mapping=column_mapping)
    report.save_html(report_path)

    result = report.as_dict()

    #price drift
    print(f"Price drift: {result['metrics'][0]['result']['drift_score']}")

    #number of drifted columns
    print(f"number of drifted columns: {result['metrics'][1]['result']['number_of_drifted_columns']}")

    #share of missing values
    print(f"share of missing values: {result['metrics'][2]['result']['current']['share_of_missing_values']}")

def transition_model(mlflow_client, model_version, new_stage, archive_existing:bool=False):
    transition_date = datetime.today().date()

    mlflow_client.transition_model_version_stage(
        name=MLFLOW_MODEL_NAME,
        version=model_version,
        stage=new_stage,
        archive_existing_versions=archive_existing
    )
    mlflow_client.update_model_version(
        name=MLFLOW_MODEL_NAME,
        version=model_version,
        description=f"The model version {model_version} was transitioned to {new_stage} on {transition_date}"
    )

def test_model(mlflow_client, test_path):
    report_type = "Test"
    #best_run_id, best_rsme, best_stage = get_best_run(mlflow_client)
    best_run_id, _, _ = get_best_run(mlflow_client)
    filter_string = f"run_id='{best_run_id}'"
    results = mlflow_client.search_model_versions(filter_string)

    if results[0]:
        model = results[0]
        new_stage = "Production"
        model_uri = f"runs:/{best_run_id}/models_mlflow"
        # just for this project, promote models for testing to stage production
        if model.current_stage == "Staging":
            transition_model(mlflow_client, model.version, new_stage, False)
        
        # load model
        booster = mlflow.xgboost.load_model(model_uri)

        df_train = pd.read_csv(TRAIN_PATH)
        df_test = pd.read_csv(test_path)
        X_train, X_test, y_train, y_test, _ = add_features(df_train, df_test)
        test = xgb.DMatrix(X_test, label=y_test)

        y_pred = booster.predict(test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print("rmse:", rmse)
        #mlflow.log_metric("rmse", rmse)
        r2 = r2_score(y_test, y_pred)
        #mlflow.log_metric("r2score", r2)
        print("r2score:", r2)

        train = xgb.DMatrix(X_train, label=y_train)
        test = xgb.DMatrix(X_test, label=y_test)
        monitor_model(report_type, pd.read_csv(TRAIN_PATH), pd.read_csv(test_path), booster, train, test)
    else:
        print("There is no model. Run training first!")

def predict(mlflow_client, dataframe):
    # load best model --> ensure that there is one
    best_run_id, best_rsme, best_stage = get_best_run(mlflow_client)

    logged_model = f"runs:/{best_run_id}/models_mlflow"

    # Load model as a XGBoostModel.
    loaded_model = mlflow.xgboost.load_model(logged_model)

    df_train = pd.read_csv(TRAIN_PATH)
    X_train, X_pred, y_train, y_pred, _ = add_features(df_train, dataframe)
    
    pred = xgb.DMatrix(X_pred, label=y_pred)

    # Predict on a Pandas DataFrame.
    result = loaded_model.predict(pred)
    return result[0]

@flow(log_prints=True)
def main_flow() -> None:
    """The main training pipeline"""
    # Preparation steps
    setup()
    
    # MLflow settings
    mlflow_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    except MlflowException:
        mlflow.create_experiment(name=MLFLOW_EXPERIMENT_NAME)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Print info about the best run
    #best_run_id, best_rsme, best_stage = get_best_run(mlflow_client)
    #print(f"best run id: {best_run_id}")
    #print(f"best rsme: {best_rsme}")
    #print(f"stage: {best_stage}")

    # Create train, val, test datasets
    if not os.path.isfile(FULL_TRAIN_PATH):    
        create_processed_dataset(category="A", train_path=FULL_TRAIN_PATH, val_path=FULL_VAL_PATH, test_path=FULL_TEST_PATH)
        create_processed_dataset(category="H", train_path=TRAIN_PATH, val_path=VAL_PATH, test_path=TEST_PATH)
        create_processed_dataset(category="C", train_path=PROBLEM_TRAIN_PATH, val_path=PROBLEM_VAL_PATH, test_path=PROBLEM_TEST_PATH)

    # Read data into DataFrame
    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)

    # Transform
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)

    # Train
    train_best_model(X_train, X_val, y_train, y_val, dv)

    # Register the model
    register_model(mlflow_client)

    # Promote the model
    promote_model(mlflow_client)

    # Test the model
    test_model(mlflow_client, TEST_PATH)

    # Predict
    # PROBLEM_TEST set has 3966 rows
    df_pred = pd.read_csv(PROBLEM_TEST_PATH)
    random_value = random.randint(0, 3965)
    dataframe = df_pred.iloc[[random_value]]
    result = predict(mlflow_client, dataframe)
    #print(dataframe[FEATURES])
    print(f"Predicted house price: {result}") 

if __name__ == "__main__":
    main_flow()
