import pickle
import pathlib
from datetime import date

import mlflow
import pandas as pd
import xgboost as xgb
from prefect import flow, task
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.metrics import r2_score, mean_squared_error
from mlflow.exceptions import MlflowException
from prefect.artifacts import create_markdown_artifact
from mlflow.data.pandas_dataset import PandasDataset
from sklearn.feature_extraction import DictVectorizer

import monitor
import variables as v


@flow(name='training_best_run_flow', log_prints=True)
def get_best_run(mlflow_client):
    """Return the run_id, rsme, and stage of the best run"""

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
            if version.current_stage == "Staging":
                if (best_stage == "") | (best_stage == "None"):
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


@task(name='training_task_preparation')
def prep_for_train(train_path, val_path, features, target):
    """Preparation for the real training step.
    Return DictVectorizer and other information for
    training and validation.
    """

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    # features = FEATURES
    # df_train[CATEGORICAL] = df_train[CATEGORICAL].astype(str)
    # df_val[CATEGORICAL] = df_val[CATEGORICAL].astype(str)

    dv = DictVectorizer()

    train_dicts = df_train[features].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[features].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train = df_train[target].values
    y_val = df_val[target].values

    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    return dv, train_path, train, val_path, valid, y_val


@flow(name='training_flow_train')
def train_model(
    mlflow_client,
    params,
    dv,
    train_path,
    train,
    val_path,
    valid,
    y_val,
    model_num_boost_round=10,
    model_early_stopping_rounds=10,
):
    """Train the xgboost-model with parameters of the preparation step."""
    report_type = "Train"
    df = pd.read_csv(train_path)
    dataset: PandasDataset = mlflow.data.from_pandas(df, source=train_path)
    if params == "":
        params = v.BEST_PARAMS

    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        # Experimental: This function may change or be removed in a future release without warning.
        mlflow.log_input(dataset, context="training")

        mlflow.log_param("train-data-path", train_path)
        mlflow.log_param("valid-data-path", val_path)
        mlflow.log_params(params)

        booster = xgb.train(
            params=params,
            dtrain=train,
            # num_boost_round=100,
            num_boost_round=model_num_boost_round,
            evals=[(valid, "validation")],
            # early_stopping_rounds=20,
            early_stopping_rounds=model_early_stopping_rounds,
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
        r2 = r2_score(y_val, y_pred)
        mlflow.log_metric("r2score", r2)

        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/dict_vect.bin", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/dict_vect.bin", artifact_path="dict_vectorizer")

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

        run = mlflow_client.search_runs(
            experiment_ids='1',
            filter_string="",
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=1,
        )
        run_id = run[0].info.run_id

        monitor.monitor_model(
            report_type,
            pd.read_csv(train_path),
            pd.read_csv(val_path),
            run_id,
            dv,
            train,
            valid,
        )

        return report_type, run_id, dv, train, valid


@flow(name='training_flow', log_prints=True)
def training_flow() -> None:
    """The main training pipeline"""

    # MLflow settings
    mlflow_client = MlflowClient(tracking_uri=v.MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(v.MLFLOW_TRACKING_URI)

    try:
        mlflow.get_experiment_by_name(v.MLFLOW_EXPERIMENT_NAME)
    except MlflowException:
        mlflow.create_experiment(name=v.MLFLOW_EXPERIMENT_NAME)

    mlflow.set_experiment(v.MLFLOW_EXPERIMENT_NAME)

    params = v.BEST_PARAMS

    dv, train_path, train, val_path, valid, y_val = prep_for_train(
        v.TRAIN_PATH, v.VAL_PATH, v.FEATURES, v.TARGET_FEATURE
    )

    train_model(
        mlflow_client, params, dv, train_path, train, val_path, valid, y_val, 10, 10
    )
