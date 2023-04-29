from sklearn import datasets
import pandas as pd
import numpy as np



def load_data(output_path: str):
    """ Function load init dataset.
    :param output_path: Path to save cleaned DataFrame
    :return:
    """
    iris = datasets.load_iris()
    iris_df = pd.DataFrame(
        data=np.c_[iris['data'], iris['target']],
        columns=iris['feature_names'] + ['target']
    )
    iris_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    load_data()
