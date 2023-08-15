# Housing-prices-France-35

## Problem description
The used data set is from Kaggle (https://www.kaggle.com/datasets/cheneblanc/housing-prices-35-fr?resource=download). The data contains information about housing prices of the departement of Ille-et-Vilaine in France from 2014 to present.
There are 10 columns:
- date  =   selling date
- position_wgs  =   House position in WGS coordinates
- x_lbt93   =   x coordinate in Lambert 93 CRS
- y_lbt93   =   y coordinate in Lambert 93 CRS
- category  =   Housing category. H: house, C: condo
- area_living   =  Area of the living area in square meters
- area_land =   Area of land for house in square meters
- n_rooms   =   Number of main rooms
- shape_wgs =   Shape of the associated land the building is on
- price     =   Price of the house

This model tries to predict the housing prices (column "price") for this data set.

## Preliminary remarks
- the data (housing-prices-35.csv) cannot be provided by this repo, because it exceeds the maximum file size to upload it to this repository --> but there is an archive.zip that contains the csv file --> it's automatically unzipped while running make prerequisites

## Workflow orchestration
The **first step** of using Prefect server is done when you run **make prerequisites** --> otherwise you have to do it manually
- open new terminal
- cd to project root
- **prefect init --recipe local** with activated virtual environment OR
- **pipenv run prefect init --recipe local**
(in the following steps I only write the first version of the commands)

For the deployment of the flow i've adapted **prefect.yaml** file.

**Second step** Start Prefect server
- open new terminal
- **prefect server start**

**Third step** Start MLFlow server
- open new terminal
- **make start_mlflow**

**Fourth step** Create work pool, deploy the flow and run the worker
- open new terminal
- **prefect work-pool delete 'zoompool'** (uncomment, if there is already a pool with this name)
- **prefect work-pool create --type process zoompool**
- **prefect --no-prompt deploy --all**
- **prefect worker start -p zoompool**
- go to prefect website (http://127.0.0.1:4200/deployments) --> "Quick run"

## Reproducibility
- run **make prerequisites** to initialize the project
    * provide virtual environment .venv in project folder
    * unzips the archive.zip to ./data/raw/housing-prices-35.csv
    * the step "Installing dependencies from Pipfile.lock" needs some time
    * initialize Prefect

- open seperate terminal and run **make start_mlflow**
    * starts MLflow server

- open seperate terminal and run **make start_prefect**
    * starts Prefect server

- open seperate terminal and run **make start_deploy**
    * starts the workflow deployment with Prefect

- open seperate terminal and run **make start_train**
    * starts the workflow

- run **make clean** to clean the project environment
    * removes the folder of the virtual environment and cleans the project folder