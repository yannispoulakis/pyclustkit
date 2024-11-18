import numpy as np
import pandas as pd


def instances_to_features_ratio(df):
    return df.shape[0] / df.shape[1]


def log_instances_to_features_ratio(df):
    return np.log((df.shape[0] / df.shape[1]))


def features_to_instances_ratio(df):
    return df.shape[0] / df.shape[1]


def log_features_to_instances_ratio(df):
    return np.log((df.shape[0] / df.shape[1]))


def pct_of_discrete(df):
    df = pd.DataFrame(df)
    int_columns = [col for col in df.columns if df[col].dtype in ['int64', 'int32', 'int']]
    return len(int_columns) / df.shape[1]


def pct_of_mv(df):
    """
    return the percentage of missing values in the dataset.
    Args:
        df:

    Returns:
        (float) in the range of (0,1)
    """
    return np.sum(np.sum(np.isnan(df))) / df.size





def log2_instances(df):
    return np.log2(df.shape[0])


def log2_attributes(df):
    return np.log2(df.shape[1])


def log10_attributes(df):
    return np.log(df.shape[1])


def log10_instances(df):
    return np.log(df.shape[0])
