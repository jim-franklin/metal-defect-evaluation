"""
Data loader for the UCI Steel Plates Faults dataset.

Dataset:
https://archive.ics.uci.edu/dataset/198/steel+plates+faults
"""

from ucimlrepo import fetch_ucirepo
import pandas as pd


def load_steel_plates_faults(return_metadata=False):
    """
    Load the Steel Plates Faults dataset.

    Parameters
    ----------
    return_metadata : bool, default=False
        If True, also returns metadata and variable information.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix.

    y : pandas.DataFrame
        Target labels.

    metadata : dict, optional

    variables : pandas.DataFrame, optional
    """

    dataset = fetch_ucirepo(id=198)

    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    if return_metadata:
        return X, y, dataset.metadata, dataset.variables

    return X, y


def load_full_dataset():
    """
    Returns features and targets combined into a single DataFrame.
    """

    X, y = load_steel_plates_faults()
    return pd.concat([X, y], axis=1)


def get_feature_names():
    """
    Returns a list of feature names.
    """

    X, _ = load_steel_plates_faults()
    return X.columns.tolist()


def get_target_names():
    """
    Returns a list of target column names.
    """

    _, y = load_steel_plates_faults()
    return y.columns.tolist()


if __name__ == "__main__":

    X, y, metadata, variables = load_steel_plates_faults(
        return_metadata=True
    )

    print(f"Features shape : {X.shape}")
    print(f"Targets shape  : {y.shape}")

    print("\nMetadata")
    print(metadata)

    print("\nVariables")
    print(variables.head())