import pandas as pd
from scipy.stats import chi2, skew
from scipy import linalg
import numpy as np


def tsquared_transform(X: np.array):
    """Calculates the t_squared transformed vector."""

    # Calculate mean vector
    mean_vector = np.mean(X, axis=0)

    # Calculate covariance matrix
    covariance_matrix = np.cov(X, rowvar=False)

    # Calculate the inverse of the covariance matrix
    inv_cov_matrix = np.linalg.inv(covariance_matrix)

    # Calculate T-squared values for each observation
    tsquared_values = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        diff = X[i, :] - mean_vector
        tsquared_values[i] = np.dot(diff, np.dot(inv_cov_matrix, diff))

    return tsquared_values


def mv_normality_test(df):
    """Calculates the percentage of values in the t_squared vector that are less than the critical value of the chi-squared
    distribution with degrees of freedom equal to the array's number of instances."""

    tsquared_vector = tsquared_transform(np.array(df))
    crit_val = chi2.ppf(0.5, df.shape[1])
    return len(np.where(tsquared_vector < crit_val)[0]) / df.shape[0]


def t_squared_skewness(df):
    """calculates the t_squared vector's skewness.
    """
    tsquared_vector = tsquared_transform(np.array(df))
    return skew(tsquared_vector)


def t_squared_outliers(df):
    """Calculates the percentage of outliers according to the t_squared vector. In such case outliers are the values
    that are more than two standard deviations distant from the mean."""
    tsquared_vector = tsquared_transform(np.array(df))
    tsquared_vector_mean = np.mean(tsquared_vector)
    tsquared_vector_std = np.std(tsquared_vector)
    outliers = np.where(tsquared_vector - tsquared_vector_mean > (2* tsquared_vector_std))
    return len(outliers)/len(tsquared_vector)