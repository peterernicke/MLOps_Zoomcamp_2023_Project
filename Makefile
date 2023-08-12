export PIPENV_VENV_IN_PROJECT := 1
export PIPENV_VERBOSITY := -1

requirements:
	python3 -m pip install --upgrade pip
	pip install --upgrade pipenv
	pipenv install --python 3.11

start_env:
	pipenv shell

code:
	black .
	isort .

start_prefect:
	pipenv run prefect server start

start_mlflow:
	pipenv run mlflow ui --backend-store-uri sqlite:///mlflow.db

start_train:
	pipenv run python orchestrate.py

clean:
	rm -rf __pycache__
	pipenv --rm