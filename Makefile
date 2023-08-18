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

start_mlflow:
	pipenv run mlflow ui --backend-store-uri sqlite:///mlflow.db

start_prefect:
	pipenv run prefect server start

start_deploy:
	# if error: Work pool named 'zoompool' already exists. 
	# Please try creating your work pool again with a different name.
	# uncomment next line
	pipenv run prefect work-pool delete 'zoompool'
	pipenv run prefect work-pool create --type process zoompool
	pipenv run prefect --no-prompt deploy --all
	pipenv run prefect worker start -p zoompool

start_train: ./data/raw/housing-prices-35.csv
	pipenv run python orchestrate.py

start_monitoring:
	pipenv run docker-compose -f ./monitoring/docker-compose.yaml up --build

#start_monitoring2:
#	pipenv run python ./monitoring/evidently_metrics_calculation.py
#   --> is not working... open new terminal and type
#	cd monitoring
#	python evidently_metrics_calculation.py

clean:
	rm -rf __pycache__
	rm -rf data/processed
	rm -rf data/raw/housing-prices-35.csv
	rm -rf evidently
	rm -rf mlruns
	rm -rf mlflow.db
	pipenv --rm