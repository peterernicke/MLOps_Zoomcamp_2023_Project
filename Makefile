export PIPENV_VENV_IN_PROJECT := 1
export PIPENV_VERBOSITY := -1

prerequisites:
	@echo "Building Python environment and unzipping dataset"
	python3 -m pip install --upgrade pip
	pip install --upgrade pipenv
	pipenv install --python 3.11
	pipenv run python ./scripts/unzipZipRaw.py

mlflow:
	@echo "Running mlflow ui"
	pipenv run mlflow ui --backend-store-uri sqlite:///mlflow.db
	#pipenv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root=s3://name-of-bucket

prefect:
	@echo "Starting Prefect server"
	pipenv run prefect server start

deploy: ./data/raw/housing-prices-35.csv
	@echo "Starting workflow deployment with Prefect"
	# if error: Work pool named 'zoompool' already exists. 
	# Please try creating your work pool again with a different name.
	# uncomment next line
	#pipenv run prefect work-pool delete 'zoompool'
	pipenv run prefect work-pool create --type process zoompool
	pipenv run prefect --no-prompt deploy --all
	pipenv run prefect worker start -p zoompool

monitoring:
	@echo "Starting monitoring with Evidently and Grafana dashboards"
	pipenv run docker-compose -f ./monitoring/docker-compose.yaml up --build
	@echo "Open a new terminal and run"
	@echo "cd monitoring"
	@echo "python evidently_metrics_calculation.py"

#monitoring2:
#	pipenv run python ./monitoring/evidently_metrics_calculation.py
#   --> is not working... open new terminal and type
#	cd monitoring
#	python evidently_metrics_calculation.py

web-service:
	@echo "Creating docker container for model deployment (as web service)"
	pipenv run docker build -f ./web-service/Dockerfile -t housing-price-prediction-service:v1
	@echo "Open a new terminal and run"
	@echo "cd web-service"
	@echo "docker run -it --rm -p 9696:9696 housing-price-prediction-service:v1"
	@echo "Open a new terminal and run"
	@echo "python test.py"
	@echo "To stop all running docker containers run"
	@echo "docker stop $(docker ps -a -q)"

# cd web-service
# docker build -t housing-price-prediction-service:v1 .
# docker run -it --rm -p 9696:9696 housing-price-prediction-service:v1
# python test.py
# docker stop $(docker ps -a -q)

clean:
	@echo "Cleaning"
	rm -rf __pycache__
	rm -rf data/processed
	rm -rf data/raw/housing-prices-35.csv
	rm -rf evidently
	rm -rf mlruns
	rm -rf mlflow.db
	pipenv --rm

code:
	@echo "Code formatting with black, isort, and pylint"
	black .
	isort .
	pylint --recursive=y .

start_env:
	pipenv shell

train: ./data/raw/housing-prices-35.csv
	@echo "Starting training"
	pipenv run python orchestrate.py