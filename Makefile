export PIPENV_VENV_IN_PROJECT := 1
export PIPENV_VERBOSITY := -1

prerequisites:
	python3 -m pip install --upgrade pip
	pip install --upgrade pipenv
	pipenv install --python 3.11
	pipenv run python ./scripts/unzipZipRaw.py

start_env:
	pipenv shell

code:
	black .
	isort .

start_prefect:
	pipenv run prefect server start

start_mlflow:
	pipenv run mlflow ui --backend-store-uri sqlite:///mlflow.db

start_train: ./data/raw/housing-prices-35.csv
	pipenv run python orchestrate.py

clean:
	rm -rf __pycache__
	pipenv --rm