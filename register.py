import mlflow
from prefect import flow
from mlflow.entities import ViewType

# from mlflow.tracking import MlflowClient

import variables as v


@flow(name='register_model_flow', log_prints=True)
def register_model(mlflow_client):
    """Register the latest run as model"""

    run = mlflow_client.search_runs(
        experiment_ids='1',
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
    )
    run_id = run[0].info.run_id
    model_uri = f"runs:/{run_id}/models_mlflow"

    registered_models = mlflow.search_registered_models()

    if not registered_models:
        print("register model")
        mlflow.register_model(model_uri=model_uri, name=v.MLFLOW_MODEL_NAME)
    else:
        try:
            for model in registered_models:
                if model.name == v.MLFLOW_MODEL_NAME:
                    lv = model.latest_versions
                    for model_version in lv:
                        if model_version.run_id == run_id:
                            print("model is already versioned")
                            raise StopIteration()
                    print("register model")
                    mlflow.register_model(model_uri=model_uri, name=v.MLFLOW_MODEL_NAME)
        except StopIteration:
            pass
