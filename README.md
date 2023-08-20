# MLOps Zoomcamp Project - Cohort 2023
|![](/images/house.jpg)|
|:--:| 
|*[Image from https://pixabay.com](https://pixabay.com/de/photos/geld-heimat-m%C3%BCnze-anlage-gesch%C3%A4ft-2724235/)*|

## Problem description
The used dataset is from [Kaggle](https://www.kaggle.com/datasets/cheneblanc/housing-prices-35-fr?resource=download). The data contains information about housing prices of the departement of Ille-et-Vilaine in France from 2014 to present.
There are 10 columns:

<div align="center">

| field | description |
|------|------------| 
| date  |selling date |
|position_wgs|House position in WGS coordinates|
|x_lbt93 |  x coordinate in Lambert 93 CRS |
|y_lbt93   |   y coordinate in Lambert 93 CRS |
|category  |  Housing category. H: house, C: condo |
|area_living   |  Area of the living area in square meters |
|area_land |  Area of land for house in square meters |
|n_rooms   |   Number of main rooms |
|shape_wgs |   Shape of the associated land the building is on |
|price     |  Price of the house |

</div>

This model tries to predict the housing price (column "price") for this data set.

## Dataset

|id| 	date |	position_wgs |	x_lbt93 |	y_lbt93 |	category |	area_living |	area_land |	n_rooms 	|shape_wgs 	|price|
|-|-------------|--------------|---------|--------|-----|--------|---------|-----------------|-------------|---------------|
|0 |	2020-09-16 |	POINT (-1.667233132948424 48.10518446058921) |	352812.796362 |	6.788576e+06 |	C |	30.0 |	0.0 |	1 |	MULTIPOLYGON (((-1.667524599999999 48.10492969... |	120000.0|
|1 |	2020-08-05 |	POINT (-1.580536596330411 48.15039836267523) |	359546.007545 |	6.793215e+06 |	C |	67.0 |	0.0 |	3 |	MULTIPOLYGON (((-1.581112699999998 48.15012009... |	176000.0|
|2 |	2020-01-06 |	POINT (-1.857796113215682 48.59007145768236) |	341968.217653 |	6.843224e+06 |	H |	56.0 |	2168.0 |	3 |	MULTIPOLYGON (((-1.857759199999999 48.58984439... |	125000.0|
|3 |	2020-01-06 |	POINT (-1.857796113215682 48.59007145768236) |	341968.217653 |	6.843224e+06 |	H |	56.0 |	2168.0 |	3 |	MULTIPOLYGON (((-1.857759199999999 48.58984439... |	125000.0|
|4 |	2020-05-19 |	POINT (-1.112155586336285 48.40433431446166) |	395794.852753 |	6.819483e+06 |	H |	32.0 |	891.0 |	1 |	MULTIPOLYGON (((-1.111970599999998 48.40436839... |	13000.0|


## Preliminary remarks
The data (data/raw/housing-prices-35.csv) cannot be provided by this repo, because it exceeds the maximum file size to upload it to this repository. But there is an archive.zip in the same folder, that contains the csv file. It's automatically unzipped while running **make prerequisites**

I recommend to run steps as it's described in Makefile.

## 1. Prerequisites
To prepare the environment just run **make prerequisites**
```
prerequisites:
	@echo "Building Python environment and unzipping dataset"
	python3 -m pip install --upgrade pip
	pip install --upgrade pipenv
	pipenv install --python 3.11
	pipenv run python ./scripts/unzipZipRaw.py
```
The step "Installing dependencies from Pipfile.lock" needs some time.
That provide the virtual environment **.venv** in project folder. This commands also unzips the needed data from archive.zip.

## 2. Start MLFlow UI
To start MLFlow UI open new terminal and run **make mlflow**
```
mlflow:
	@echo "Running mlflow ui"
	pipenv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
You can access the initialized GUI at http://127.0.0.1:5000.

![](/images/mlflow-initialGUI.png)



## 3. Start Prefect server
To start Prefect server open new terminal and run **make prefect**
```
prefect:
	@echo "Starting Prefect server"
	pipenv run prefect server start
```
You can access the initialized GUI at http://127.0.0.1:4200.

![](/images/prefect-initialGUI.png)

## 4. Workflow orchestration
Now everything is ready to start the orchestration workflow. Run **make deploy** in a new terminal window.
```
deploy: ./data/raw/housing-prices-35.csv
	@echo "Starting workflow deployment with Prefect"
	# if error: Work pool named 'zoompool' already exists. 
	# Please try creating your work pool again with a different name.
	# uncomment next line
	#pipenv run prefect work-pool delete 'zoompool'
	pipenv run prefect work-pool create --type process zoompool
	pipenv run prefect --no-prompt deploy --all
	pipenv run prefect worker start -p zoompool
```
With that everything is prepared. You only need to go to [Prefect website](http://127.0.0.1:4200/deployments) and hit "Quick run"

![](/images/prefect-quickRun.png)

Running the flows (and sub flows) and tasks can take some time.
This workflow includes a whole bunch of steps. 
First the datasets are provided. You can find all of them in the data/processed folder. After that there is a preparation step. For the training I have chosen the columns "area_living", "area_land", "n_rooms", "price", and "year". I extracted the "year" information from the "date" column.
This specific features are used in the training step.
You can find the run on MLFlow website. The model is now registered and I also promoted it to Staging stage automatically.

![](/images/mlflow-Run.png)

After that a test is triggered. For this purpose I decided to transition that model to Production. This decision can make no sense in production environment but I wanted to see this transition in action.

![](/images/mlflow-prodModel.png)

As the last step there is a prediction. For that purpose I used a different dataset. The dataset has almost 4.000 rows and for the prediction a randomly choosen row is used. You can find the predicted price in the terminal.

All of this mentioned steps are shown in the Prefect GUI after the main flow has finished.

![](/images/prefect-FlowCompleted.png)

For the training and the test an automatically created HTML report is stored in the evidently folder.

![](/images/report.png)

## 5. Monitoring with Evidently and Grafana
This step shows Evidently and Grafana in action. It is dockerized (have a look at monitoring/docker-compose.yaml). To start this step open new terminal and run **make monitoring**. Run this make command in root directory.
```
monitoring:
	@echo "Starting monitoring with Evidently and Grafana dashboards"
	pipenv run docker-compose -f ./monitoring/docker-compose.yaml up --build
	@echo "Open a new terminal and run"
	@echo "cd monitoring"
	@echo "python evidently_metrics_calculation.py"
```
This provides 3 running docker containers for you (database, [Grafana](http://localhost:3000/), and [Adminer](http://localhost:8080/)). The user credentials for Grafana are admin:admin.
Then you have to open new terminal and change directory to the monitoring folder and run **python evidently_metrics_calculation.py** manually.
A process is triggered to simulate production usage of the model. For that purpose some metrics are calculated for 9 runs (3 for each of 3 different data sets). On Grafana website you can find a prepared dashboard "Housing Prices Prediction Dashboard". After finishing you can see the results.

![](/images/grafana-db.png)

I implemented also simple alerting to raise an error when one specific value is higher than expected.  

![](/images/grafana-alerting.png)


## 6. Model deployment as simple web service
This step is about deploying the model as a web service. It is also dockerized (have a look at the Dockerfile in the web-service folder). The image building process can be triggered by running **make web-service**.
```
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
```
Then you have to change directory to the web-service folder. By running **docker run -it --rm -p 9696:9696 housing-price-prediction-service:v1** the docker container is started. The web service is listening at http://0.0.0.0:9696. 
Open a new terminal (in the web-service folder) and run **python test.py**. This triggers a request to get a prediction for one specific example. This triggers one request and outputs the result of the prediction to the terminal.

## 7. Cleaning
To clean everything open new terminal and run **make clean**
```
clean:
	@echo "Cleaning"
	rm -rf __pycache__
	rm -rf data/processed
	rm -rf data/raw/housing-prices-35.csv
	rm -rf evidently
	rm -rf mlruns
	rm -rf mlflow.db
	pipenv --rm
```
This also removes the virtual environment in the project folder **.venv**


## Reproducibility
Following each step in the mentioned order should make it easy to reproduce the same results like me.

## Best Practices
There are unit tests implemented and I used black, isort and pylint as linter and code formatters (have a look at pyproject.toml). I used a Makefile for the most important steps. This order should guide you through the project. I also implemented pre-commit hooks (see .pre-commit-config.yaml) and I added ci-tests (see .github/workflows/ci-tests.yml).