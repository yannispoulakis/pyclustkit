"""
A collection of methods for various meta-features that fall into the descriptive category.
"""
import numpy as np
import pandas as pd


def instances_to_features_ratio(df):
    """
    Calculates the instances to features ratio.
    Args:
        df (np.ndarray or pd.DataFrame): The data

    Returns:
        float: The ratio of instances to features.
    """
    return df.shape[0] / df.shape[1]


def log_instances_to_features_ratio(df):
    """
    Calculates the logarithm of the instances to features ratio.
    Args:
        df (np.ndarray or pd.DataFrame): The data

    Returns:
        float: The logarithm of the ratio of instances to features.
    """
    return np.log((df.shape[0] / df.shape[1]))


def features_to_instances_ratio(df):
    """
    Calculates the features to instances ratio.
    Args:
        df (np.ndarray or pd.DataFrame): The data

    Returns:
        float: The features to instances ratio.
    """
    return df.shape[0] / df.shape[1]


def log_features_to_instances_ratio(df):
    """
    Calculates the logarithm of the features to instances ratio.
    Args:
        df (np.ndarray or pd.DataFrame): The data

    Returns:
        float: The logarithm of the features to instances ratio.
    """
    return np.log((df.shape[0] / df.shape[1]))


def pct_of_discrete(df):
    """
    A method to find the percentage of columns that contain discrete values. Such columns are identified automatically

    Args:
        df (np.ndarray or pd.DataFrame):  The dataset

    Returns:
        float: percentage of columns with discrete values in the dataset
    """
    df = pd.DataFrame(df)
    int_columns = [col for col in df.columns if df[col].dtype in ['int64', 'int32', 'int']]
    return len(int_columns) / df.shape[1]


def pct_of_mv(df):
    """
    return the percentage of missing values in the dataset.
    Args:
        df (np.nd.array or pd.DataFrame): The dataset.

    Returns:
        float: in the range of (0,1)
    """
    return np.sum(np.sum(np.isnan(df))) / df.size


def log2_instances(df):
    """
    Calculates the logarithm with base 2 of the number of instances present in the dataset
    Args:
        df (np.nd.array or pd.DataFrame): The dataset.

    Returns:
        float: log2 of the number of instances in dataset.
    """
    return np.log2(df.shape[0])


def log10_instances(df):
    """
    Calculates the logarithm with base 10 of the number of instances present in the dataset
    Args:
        df (np.nd.array or pd.DataFrame): The dataset.

    Returns:
        float: log10 of the number of instances in dataset.
    """
    return np.log(df.shape[0])


def log2_attributes(df):
    """
    Calculates the logarithm with base 2 of the number of features present in the dataset
    Args:
        df (np.nd.array or pd.DataFrame): The dataset.

    Returns:
        float: log2 of the number of features in dataset.
    """
    return np.log2(df.shape[1])


def log10_attributes(df):
    """
    Calculates the logarithm with base 10 of the number of instances present in the dataset
    Args:
        df (np.nd.array or pd.DataFrame): The dataset.

    Returns:
        float: log10 of the number of instances in dataset.
    """
    return np.log(df.shape[1])


