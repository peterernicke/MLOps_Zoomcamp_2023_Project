# Housing-prices-France-35

## Problem description
The used data set is from Kaggle (https://www.kaggle.com/datasets/cheneblanc/housing-prices-35-fr?resource=download). The data contains information about housing prices of the departement of Ille-et-Vilaine in France from 2014 to present.
There are 10 columns:
- date  =   selling date
- position_wgs  =   House position in WGS coordinates
- x_lbt93   =   x coordinate in Lambert 93 CRS
- y_lbt93   =   y coordinate in Lambert 93 CRS
- category  =   Housing category. H: house, C: condo
- area_living   =  Area of the living are in square meters 
- area_land =   Area of land for house in square meters
- n_rooms   =   Number of main rooms
- shape_wgs =   Shape of the associated land the building is on
- price     =   Price of the house

This model tries to predict the housing prices (column "price") for this data set.

## Preliminary remarks
- the data (housing-prices-35.csv) cannot be provided by this repo, because it exceeds the maximum file size to upload it to this repository --> but there is an archive.zip that contains the csv file --> it's automatically unzipped while running make prerequisites

## Reproducibility
- run **make prerequisites** to initialize the project
    * provide virtual environment .venv in project folder
    * unzips the archive.zip to ./data/raw/housing-prices-35.csv
    * the step "Installing dependencies from Pipfile.lock" needs some time

- open seperate terminal and run **make start_mlflow**
    * starts MLflow server

- open seperate terminal and run **make start_prefect**
    * starts Prefect server

- open seperate terminal and run **make start_train**
    * starts the workflow

- run **make clean** to clean the project environment
    * removes the folder of the virtual environment 