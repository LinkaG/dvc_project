import joblib
import click
from sklearn import metrics
import pandas as pd
import json
from ruamel.yaml import YAML
import mlflow



@click.command()
@click.argument("input_path_data", type=click.Path())
@click.argument("input_path_label", type=click.Path())
@click.argument("model_path", type=click.Path())
@click.argument("output_path", type=click.Path())
def evaluate(input_path_data: str, input_path_label: str, model_path: str, output_path: str):
    mlflow.set_experiment("logreg classifier")


    yaml = YAML(typ="safe")
    params = yaml.load(open("params.yaml", encoding="utf-8"))['model']['params']

    X_test = pd.read_csv(input_path_data)
    y_test = pd.read_csv(input_path_label)
    clf = joblib.load(model_path)
    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)

    score = dict(
        accuracy=float(accuracy)
    )
    with open(output_path, 'w') as score_file:
        json.dump(score, score_file, indent=4)

    mlflow.log_params(params)
    mlflow.log_metrics(score)


if __name__ == "__main__":
    evaluate()