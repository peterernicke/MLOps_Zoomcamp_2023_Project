from datetime import date, datetime

import mlflow
from prefect import flow
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric
)

import variables as v


@flow(name='monitoring_flow', log_prints=True)
# def monitor_model(report_type, train_data, val_data, model, dv, train, valid):
def monitor_model(report_type, train_data, val_data, run_id, dv, train, valid):
    today = datetime.now()
    today = f"{today.year}-{today.month:02d}-{today.day:02d}-{today.hour:02d}:{today.minute:02d}"
    report_name = f"Evidently{report_type}Report-{run_id}-{today}.html"
    report_path = f"{v.EVIDENTLY_REPORT_PATH}{report_name}"

    model_uri = f"runs:/{run_id}/models_mlflow"
    model = mlflow.xgboost.load_model(model_uri)

    train_preds = model.predict(train)
    train_data['prediction'] = train_preds

    val_preds = model.predict(valid)
    val_data['prediction'] = val_preds

    column_mapping = ColumnMapping(
        target=None,
        prediction='prediction',
        numerical_features=v.NUMERICAL,
        categorical_features=v.CATEGORICAL,
    )

    report = Report(
        metrics=[
            ColumnDriftMetric(column_name='price'),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
        ]
    )

    print("Running report...")
    report.run(
        reference_data=train_data, current_data=val_data, column_mapping=column_mapping
    )
    report.save_html(report_path)

    result = report.as_dict()

    # price drift
    print(f"Price drift: {result['metrics'][0]['result']['drift_score']}")

    # number of drifted columns
    print(
        f"number of drifted columns: {result['metrics'][1]['result']['number_of_drifted_columns']}"
    )

    # share of missing values
    print(
        f"share of missing values: {result['metrics'][2]['result']['current']['share_of_missing_values']}"
    )


if __name__ == "__main__":
    monitor_model(None, None, None, None, None, None, None)
