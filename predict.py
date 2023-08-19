import pickle

import mlflow
import pandas as pd
import xgboost as xgb
from prefect import flow, task
from mlflow.tracking import MlflowClient
from sklearn.metrics import r2_score, mean_squared_error

import train as training
import monitor
import promote
import variables as v


@flow(name='testing_flow', log_prints=True)
def test_model(mlflow_client, run_id, test_path, train_path):
    report_type = "Test"
    if not run_id:
        # best_run_id, best_rsme, best_stage = get_best_run(mlflow_client)
        run_id, _, _ = training.get_best_run(mlflow_client)
    filter_string = f"run_id='{run_id}'"
    results = mlflow_client.search_model_versions(filter_string)

    if results[0]:
        model = results[0]
        new_stage = "Production"
        model_uri = f"runs:/{run_id}/models_mlflow"
        ### new
        dv_uri = f"./mlruns/1/{run_id}/artifacts/dict_vectorizer/dict_vect.bin"
        with open(dv_uri, 'rb') as f_out:
            dv = pickle.load(f_out)
        ### new
        # just for this project, promote models for testing to stage production
        if model.current_stage == "Staging":
            promote.transition_model(mlflow_client, model.version, new_stage, False)

        # load model
        booster = mlflow.xgboost.load_model(model_uri)

        ### new
        df_test = pd.read_csv(test_path)
        test_dicts = df_test[v.FEATURES].to_dict(orient="records")
        X_test = dv.transform(test_dicts)
        y_test = df_test[v.TARGET_FEATURE].values
        ### new

        test = xgb.DMatrix(X_test, label=y_test)

        y_pred = booster.predict(test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print("########################################")
        print("Test statistic for model: ", run_id)
        print("rmse:", rmse)
        # mlflow.log_metric("rmse", rmse)
        r2 = r2_score(y_test, y_pred)
        # mlflow.log_metric("r2score", r2)
        print("r2score:", r2)
        print("########################################")

        df_train = pd.read_csv(train_path)
        train_dicts = df_train[v.FEATURES].to_dict(orient="records")
        X_train = dv.transform(train_dicts)
        y_train = df_train[v.TARGET_FEATURE].values
        train = xgb.DMatrix(X_train, label=y_train)
        # test = xgb.DMatrix(X_test, label=y_test)
        monitor.monitor_model(
            report_type,
            pd.read_csv(train_path),
            pd.read_csv(test_path),
            # booster,
            run_id,
            dv,
            train,
            test,
        )
    else:
        print("There is no model. Run training first!")


@flow(name='prediction_flow', log_prints=True)
def predict(mlflow_client, run_id, dataframe):  # (mlflow_client, dataframe):
    if not run_id:
        # load best model --> ensure that there is one
        # best_run_id, best_rsme, best_stage = get_best_run(mlflow_client)
        run_id, _, _ = training.get_best_run(mlflow_client)
    # filter_string = f"run_id='{run_id}'"
    # results = mlflow_client.search_model_versions(filter_string)

    ### new
    dv_uri = f"./mlruns/1/{run_id}/artifacts/dict_vectorizer/dict_vect.bin"
    with open(dv_uri, 'rb') as f_out:
        dv = pickle.load(f_out)

    # df_test = pd.read_csv(pred_path)
    df_test = dataframe
    test_dicts = df_test[v.FEATURES].to_dict(orient="records")
    X_test = dv.transform(test_dicts)
    y_test = df_test[v.TARGET_FEATURE].values
    ### new

    # load model
    model_uri = f"runs:/{run_id}/models_mlflow"
    model_uri = f"runs:/{run_id}/models_mlflow"
    loaded_model = mlflow.xgboost.load_model(model_uri)

    pred = xgb.DMatrix(X_test, label=y_test)
    result = loaded_model.predict(pred)

    return result[0]

    '''
    # load best model --> ensure that there is one
    best_run_id, best_rsme, best_stage = get_best_run(mlflow_client)

    logged_model = f"runs:/{best_run_id}/models_mlflow"

    # Load model as a XGBoostModel.
    loaded_model = mlflow.xgboost.load_model(logged_model)

    df_train = pd.read_csv(v.TRAIN_PATH)
    X_train, X_pred, y_train, y_pred, _ = add_features(df_train, dataframe)

    pred = xgb.DMatrix(X_pred, label=y_pred)

    # Predict on a Pandas DataFrame.
    result = loaded_model.predict(pred)
    return result[0]'''


if __name__ == "__main__":
    test_model(MlflowClient(tracking_uri=v.MLFLOW_TRACKING_URI), None, v.TEST_PATH)
