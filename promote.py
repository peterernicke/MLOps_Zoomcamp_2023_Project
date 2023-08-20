from datetime import datetime

# import mlflow
from prefect import flow

import train as training
import variables as v


@flow(name='promote_flow', log_prints=True)
def promote_model(mlflow_client):
    """This function only transitions the first model to Staging"""
    latest_versions = mlflow_client.get_latest_versions(name=v.MLFLOW_MODEL_NAME)

    # get the registered run with the best rsme
    best_run_id, best_rsme, _ = training.get_best_run(mlflow_client)

    if latest_versions:
        for version in latest_versions:
            run = mlflow_client.get_run(version.run_id)
            rsme = int(run.data.metrics['rmse'])
            new_stage = "Staging"

            if (rsme < best_rsme) & (version.current_stage == "None"):
                # transition model to next stage
                transition_model(mlflow_client, version.version, new_stage, False)
            elif (version.run_id == best_run_id) & (version.current_stage == "None"):
                # transition model to next stage
                transition_model(mlflow_client, version.version, new_stage, False)


@flow(name='transition_flow', log_prints=True)
def transition_model(
    mlflow_client, model_version, new_stage, archive_existing: bool = False
):
    """This function can be used generally to transition a model to another stage"""
    transition_date = datetime.today().date()

    mlflow_client.transition_model_version_stage(
        name=v.MLFLOW_MODEL_NAME,
        version=model_version,
        stage=new_stage,
        archive_existing_versions=archive_existing,
    )
    mlflow_client.update_model_version(
        name=v.MLFLOW_MODEL_NAME,
        version=model_version,
        description=f"The model version {model_version} was transitioned to {new_stage} on {transition_date}",
    )
