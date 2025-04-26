import pandas as pd
from scipy.stats import pearsonr, entropy, skew, kurtosis, chi2_contingency
import itertools
import warnings
from typing import Union
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import numpy as np


from ..metalearning.utils import find_discrete


def mean_absolute_continuous_feature_correlation(df: pd.DataFrame) -> float:
    """calculates the mean pairwise correlation of the given dataframe's features. Currently, supports only Pearson's
    correlation."""
    df = pd.DataFrame(df)
    cor_sum = 0
    continuous_cols = [col for col in df.columns if df[col].dtype in ["float"]]
    if len(continuous_cols) == 0:
        return 0
    combinations = list(itertools.combinations(continuous_cols, 2))
    for comb in combinations:
        cor_sum += np.abs(pearsonr(df.loc[:, comb[0]], df.loc[:, comb[1]])[0])
    return np.abs(cor_sum / len(combinations))


def mean_entropy_of_discrete(df: np.array) -> float:
    """calculates the mean entropy of discrete features."""
    df = pd.DataFrame(df)
    int_cols = find_discrete(df)
    if len(int_cols) != 0:
        df = df[find_discrete(df)]
        entropy_sum = 0
        for col in df.columns:
            entropy_sum += entropy(df[col])
        return entropy_sum / len(df.columns)
    else:
        return 0


# Skewness based
def mean_skewness_of_continuous(df: np.array) -> Union[dict, np.nan]:
    """calculates the mean skewness of continuous attributes"""

    warnings.warn("This function uses -skewness_vector- as a global variable to avoid multiple computations.")
    df = pd.DataFrame(df)
    continuous_cols = [col for col in df.columns if df[col].dtype in ["float"]]
    if len(continuous_cols) == 0:
        return np.nan
    if 'skewness_vector' in globals():
        pass
    else:
        global skewness_vector
        skewness_vector = []
        for col in continuous_cols:
            skewness_vector.append(skew(df[col]))
    return np.mean(skewness_vector)


def min_skewness_of_continuous(df: np.array) -> Union[dict, np.nan]:
    """calculates the min skewness of continuous attributes"""

    warnings.warn("This function uses -skewness_vector- as a global variable to avoid multiple computations.")

    df = pd.DataFrame(df)
    continuous_cols = [col for col in df.columns if df[col].dtype in ["float"]]
    if len(continuous_cols) == 0:
        return np.nan
    if 'skewness_vector' in globals():
        pass
    else:
        global skewness_vector
        skewness_vector = []
        for col in continuous_cols:
            skewness_vector.append(skew(df[col]))
    return np.min(skewness_vector)


def max_skewness_of_continuous(df: np.array) -> Union[dict, np.nan]:
    """calculates the max skewness of continuous attributes"""

    warnings.warn("This function uses -skewness_vector- as a global variable to avoid multiple computations.")
    df = pd.DataFrame(df)
    continuous_cols = [col for col in df.columns if df[col].dtype in ["float"]]
    if len(continuous_cols) == 0:
        return np.nan
    if 'skewness_vector' in globals():
        pass
    else:
        global skewness_vector
        skewness_vector = []
        for col in continuous_cols:
            skewness_vector.append(skew(df[col]))
    return np.max(skewness_vector)


def std_skewness_of_continuous(df: np.array) -> Union[dict, np.nan]:
    """calculates the std skewness of continuous attributes"""

    warnings.warn("This function uses -skewness_vector- as a global variable to avoid multiple computations.")
    df = pd.DataFrame(df)
    continuous_cols = [col for col in df.columns if df[col].dtype in ["float"]]
    if len(continuous_cols) == 0:
        return np.nan
    if 'skewness_vector' in globals():
        pass
    else:
        global skewness_vector
        skewness_vector = []
        for col in continuous_cols:
            skewness_vector.append(skew(df[col]))
    return np.std(skewness_vector)


# Kurtosis based
def std_kurtosis_of_continuous(df: np.array) -> Union[dict, np.nan]:
    """calculates the std kurtosis of continuous attributes"""

    warnings.warn("This function uses -skewness_vector- as a global variable to avoid multiple computations.")
    df = pd.DataFrame(df)
    continuous_cols = [col for col in df.columns if df[col].dtype in ["float"]]
    if len(continuous_cols) == 0:
        return np.nan
    if 'kurtosis_vector' in globals():
        pass
    else:
        global kurtosis_vector
        kurtosis_vector = []
        for col in continuous_cols:
            kurtosis_vector.append(skew(df[col]))
    return np.std(kurtosis_vector)


def min_kurtosis_of_continuous(df: np.array) -> Union[dict, np.nan]:
    """calculates the min kurtosis of continuous attributes"""

    warnings.warn("This function uses -skewness_vector- as a global variable to avoid multiple computations.")
    df = pd.DataFrame(df)
    continuous_cols = [col for col in df.columns if df[col].dtype in ["float"]]
    if len(continuous_cols) == 0:
        return np.nan
    if 'kurtosis_vector' in globals():
        pass
    else:
        global kurtosis_vector
        kurtosis_vector = []
        for col in continuous_cols:
            kurtosis_vector.append(skew(df[col]))
    return np.min(kurtosis_vector)


def max_kurtosis_of_continuous(df: np.array) -> Union[dict, np.nan]:
    """calculates the max kurtosis of continuous attributes"""

    warnings.warn("This function uses -skewness_vector- as a global variable to avoid multiple computations.")
    df = pd.DataFrame(df)
    continuous_cols = [col for col in df.columns if df[col].dtype in ["float"]]
    if len(continuous_cols) == 0:
        return np.nan
    if 'kurtosis_vector' in globals():
        pass
    else:
        global kurtosis_vector
        kurtosis_vector = []
        for col in continuous_cols:
            kurtosis_vector.append(skew(df[col]))
    return np.max(kurtosis_vector)


def mean_kurtosis_of_continuous(df: np.array) -> Union[dict, np.nan]:
    """calculates the mean kurtosis of continuous attributes"""

    warnings.warn("This function uses -skewness_vector- as a global variable to avoid multiple computations.")
    df = pd.DataFrame(df)
    continuous_cols = [col for col in df.columns if df[col].dtype in ["float"]]
    if len(continuous_cols) == 0:
        return np.nan
    if 'kurtosis_vector' in globals():
        pass
    else:
        global kurtosis_vector
        kurtosis_vector = []
        for col in continuous_cols:
            kurtosis_vector.append(skew(df[col]))
    return np.mean(kurtosis_vector)


def pct_of_outliers(df):
    """Calculates the percentage of outliers found via quantiles. """
    df = pd.DataFrame(df)
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    outdf = df[(df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))]
    return outdf.notna().any(axis=1).sum() / outdf.shape[0]


def mean_concentration_between_discrete(df):
    """

    Args:
        df:

    Returns:
        float or NaN: NaN if no categorical values are present in the data set.
    """
    discrete_cols = find_discrete(df)
    discrete_cols_combinations = itertools.combinations(discrete_cols, 2)
    return np.mean([chi2_contingency(df[x[0]], df[x[1]]) for x in discrete_cols_combinations])


def hopkins(X: np.array, portion=0.1, seed=247) -> float:
    """Implementation of the hopklins statistic to measure cluster tendency. Sample size is calculated according to
     heuristics based on : Validating Clusters using the Hopkins Statistic from IEEE 2004. If the data size is less than
     100, sample size equals to the original data's records."""
    # X: numpy array of shape (n_samples, n_features)
    n = X.shape[0]
    d = X.shape[1]
    if X.shape[0] < 100:
        m = X.shape[0]
    else:
        m = int(portion * n)

    np.random.seed(seed)
    nbrs = NearestNeighbors(n_neighbors=1).fit(X)
    # u_dist
    rand_X = np.random.uniform(X.min(axis=0), X.max(axis=0), size=(m, d))
    u_dist = nbrs.kneighbors(rand_X, return_distance=True)[0]
    # w_dist
    idx = np.random.choice(n, size=m, replace=False)
    w_dist = nbrs.kneighbors(X[idx, :], 2, return_distance=True)[0][:, 1]

    U = (u_dist ** d).sum()
    W = (w_dist ** d).sum()
    H = U / (U + W)
    return H


def pca_95_deviations_to_features_ratio(X):
    pca = PCA(.95)
    return pca.fit_transform(X).shape[1] / X.shape[1]


def skewness_of_pca_first_component(X):
    pca = PCA(n_components=X.shape[1])
    return skew(pca.fit_transform(X)[0])


def kurtosis_of_pca_first_component(X):
    pca = PCA(n_components=X.shape[1])
    return kurtosis(pca.fit_transform(X)[0])
