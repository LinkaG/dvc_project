from hydra.utils import instantiate
import click
import pandas as pd
from hydra import initialize, compose
from joblib import dump
from ruamel.yaml import YAML
import mlflow
from mlflow.models.signature import infer_signature
import os

os.environ["AWS_ACCESS_KEY_ID"] = "YCAJE2FLNfw7QmIAn2n4tPXkr"
os.environ["AWS_SECRET_ACCESS_KEY"] = "YCNDXoIGWr7pzWkBkKvx3PGKIsBpOYraswRVLRKG"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["ARTIFACT_ROOT"]="s3://dvcbucket"

yaml = YAML(typ="safe")

@click.command()
@click.argument("input_path_data", type=click.Path())
@click.argument("input_path_label", type=click.Path())
@click.argument("output_path", type=click.Path())
def train(input_path_data: str, input_path_label: str, output_path: str):
    mlflow.set_tracking_uri("http://188.227.32.199/mlflow")
    mlflow.sklearn.autolog()
    mlflow.set_experiment("svm classifier1")
    mlflow.start_run(run_name ="тренировка")

    params = yaml.load(open("params.yaml", encoding="utf-8"))['model']['params']
    X_train = pd.read_csv(input_path_data)
    y_train = pd.read_csv(input_path_label)
    #Обучение модели
    # initialize(config_path='/', version_base=None)
    # cfg = compose(config_name="params")['model']['params']
    model = instantiate(params)
    model.fit(X_train, y_train.values.ravel())
    dump(model, output_path)

    # signature = infer_signature(X_train, model.predict(X_train))
    # mlflow.sklearn.log_model(model, "svm_clf", signature=signature)

if __name__ == "__main__":
    train()