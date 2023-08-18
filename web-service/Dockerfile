#FROM python:3.9.7-slim
FROM python:3.11.3-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

#COPY [ "predict.py", "models", "./" ]

COPY [ "predict.py", "./" ]

COPY [ "models/models_mlflow/*", "./models/models_mlflow/" ]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]