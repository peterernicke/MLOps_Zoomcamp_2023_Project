import random
import os.path

import mlflow
import pandas as pd
from prefect import flow, task
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

import train as training
import monitor
import predict as prediction
import promote
import register
import variables as v


@task(name='orchestrate_task_setup', log_prints=True)
def setup():
    """Create data folder and evidently report folder"""

    if not os.path.exists(v.DATA_PATH):
        os.makedirs(v.DATA_PATH)

    if not os.path.exists(v.EVIDENTLY_REPORT_PATH):
        os.makedirs(v.EVIDENTLY_REPORT_PATH)


@task(name='orchestrate_task_create_datasets', log_prints=True)
def create_processed_dataset(
    source_csv_file: str = v.CSV_FILE,
    category: str = "",  # H-houses, C-Condo, A-All
    train_path: str = "",
    val_path: str = "",
    test_path: str = "",
):
    """Create data sets for the specific use cases (train, stress, full)"""

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


@flow(name='orchestrate_flow', log_prints=True)
def main_flow() -> None:
    """The main training pipeline"""

    # Preparation steps
    setup()

    # MLflow settings
    mlflow_client = MlflowClient(tracking_uri=v.MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(v.MLFLOW_TRACKING_URI)

    try:
        mlflow.get_experiment_by_name(v.MLFLOW_EXPERIMENT_NAME)
    except MlflowException:
        mlflow.create_experiment(name=v.MLFLOW_EXPERIMENT_NAME)

    mlflow.set_experiment(v.MLFLOW_EXPERIMENT_NAME)

    # Create train, val, test datasets
    if not os.path.isfile(v.FULL_TRAIN_PATH):
        create_processed_dataset(
            category="A",
            train_path=v.FULL_TRAIN_PATH,
            val_path=v.FULL_VAL_PATH,
            test_path=v.FULL_TEST_PATH,
        )
        create_processed_dataset(
            category="H",
            train_path=v.TRAIN_PATH,
            val_path=v.VAL_PATH,
            test_path=v.TEST_PATH,
        )
        create_processed_dataset(
            category="C",
            train_path=v.PROBLEM_TRAIN_PATH,
            val_path=v.PROBLEM_VAL_PATH,
            test_path=v.PROBLEM_TEST_PATH,
        )

    # Read data into DataFrame, transform data and provide vars for model training
    dv, train_path, train, val_path, valid, y_val = training.prep_for_train(
        v.TRAIN_PATH, v.VAL_PATH, v.FEATURES, v.TARGET_FEATURE
    )

    # Train the model
    training.train_model(
        mlflow_client, "", dv, train_path, train, val_path, valid, y_val
    )
    # report_type, run_id, dv, train, valid = training.train_model(mlflow_client, "", dv, train_path, train, val_path, valid, y_val)
    # monitor.monitor_model(
    #        report_type,
    #        pd.read_csv(train_path),
    #        pd.read_csv(val_path),
    #        run_id,
    #        dv,
    #        train,
    #        valid,
    #    )

    # Register the model
    register.register_model(mlflow_client)

    # Test best run result
    best_run_id, best_rsme, best_stage = training.get_best_run(mlflow_client)
    print(f"best run id: {best_run_id}")
    print(f"best rsme: {best_rsme}")
    print(f"stage: {best_stage}")

    # Promote the model
    promote.promote_model(mlflow_client)

    # Test the model
    prediction.test_model(mlflow_client, None, v.TEST_PATH, v.TRAIN_PATH)

    # Predict
    # PROBLEM_TEST set has 3966 rows
    df_pred = pd.read_csv(v.PROBLEM_TEST_PATH)
    random_value = random.randint(0, 3965)
    dataframe = df_pred.iloc[[random_value]]
    # result = predict(mlflow_client, dataframe)
    # print(dataframe[FEATURES])
    # print(f"Predicted house price: {result}")
    # run_id = ""
    result = prediction.predict(mlflow_client, None, dataframe)
    # result = prediction.predict(mlflow_client, run_id, dataframe)
    print(f"Predicted house price: {result}")

    # Retrain the model
    # print("Retrain the model")
    # dv, train_path, train, val_path, valid, y_val = training.prep_for_train(v.PROBLEM_TRAIN_PATH, v.VAL_PATH, v.FEATURES, v.TARGET_FEATURE)
    # training.train_model(mlflow_client, "", dv, train_path, train, val_path, valid, y_val, 12, 12)
    # Test new model
    # prediction.test_model(mlflow_client, None, v.TEST_PATH, v.PROBLEM_TRAIN_PATH)


if __name__ == "__main__":
    main_flow()
