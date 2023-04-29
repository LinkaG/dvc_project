from hydra.utils import instantiate
import click
import pandas as pd
from hydra import initialize, compose
from joblib import dump
from ruamel.yaml import YAML

yaml = YAML(typ="safe")

@click.command()
@click.argument("input_path_data", type=click.Path())
@click.argument("input_path_label", type=click.Path())
@click.argument("output_path", type=click.Path())
def train(input_path_data: str, input_path_label: str, output_path: str):
    params = yaml.load(open("params.yaml", encoding="utf-8"))['model']['params']
    X_train = pd.read_csv(input_path_data)
    y_train = pd.read_csv(input_path_label)
    #Обучение модели
    # initialize(config_path='/', version_base=None)
    # cfg = compose(config_name="params")['model']['params']
    model = instantiate(params)
    model.fit(X_train, y_train.values.ravel())
    dump(model, output_path)


if __name__ == "__main__":
    train()