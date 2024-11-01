from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import zscore
from scipy.stats import spearmanr

def distance_vector(X):
    dvector = []
    for i in range(X.shape[0]):
        for j in range(i + 1, X.shape[0]):
            dvector.append(np.linalg.norm(X[i, :] - X[j, :]))
    return np.asarray(dvector)


def correlation_vector(X):
    cvector = []
    for i in range(X.shape[0]):
        for j in range(i + 1, X.shape[0]):
            cor_ = spearmanr(X[i, :], X[j, :]).correlation
            try:
                cor_ = float(cor_)
            except:
                cor_ = 0
            cvector.append(cor_)
    return np.asarray(cvector)








class CaD:
    # TODO: Check if the meta features need to be normalized
    def __init__(self):
        self.df = None
        self.distance_vector = None
        self.correlation_vector = None
        self.ferrari_distance_mf = None
        self.correlation_mf = None
        pass

    @staticmethod
    def _calculate_vector_properties(vector):
        """
        This method is used to calculate the meta-features over the pairwise distances or the pairwise correlation
        vectors. In general any other vector related to the dataset can be used.

        :param vector: A vector containing some form of pairwise instance distances.
        :type vector: list or np.ndarray
        :return: Meta-Features extracted over the instance vector.
        :rtype: dict
        """
        mf = dict()

        # statistical properties
        mf['Mean'] = np.mean(vector)
        mf['Variance'] = np.var(vector)
        mf['Standard deviation'] = np.std(vector)
        mf['Skewness'] = skew(vector)[0]
        mf['Kurtosis'] = kurtosis(vector)[0]

        # Distances in certain ranges of [0,1]
        mf['% of values in [0, 0.1]'] = np.count_nonzero(vector <= 0.1) / vector.shape[0]
        mf['% of values in [0.1, 0.2]'] = (np.count_nonzero(np.logical_and(0.1 < vector, vector <= 0.2)) /
                                           vector.shape[0])
        mf['% of values in [0.2, 0.3]'] = (np.count_nonzero(np.logical_and(0.2 < vector, vector <= 0.3)) /
                                           vector.shape[0])
        mf['% of values in [0.3, 0.4]'] = (np.count_nonzero(np.logical_and(0.3 < vector, vector <= 0.4)) /
                                           vector.shape[0])
        mf['% of values in [0.4, 0.5]'] = (np.count_nonzero(np.logical_and(0.4 < vector, vector <= 0.5)) /
                                           vector.shape[0])
        mf['% of values in [0.5, 0.6]'] = (np.count_nonzero(np.logical_and(0.5 < vector, vector <= 0.6)) /
                                           vector.shape[0])
        mf['% of values in [0.6, 0.7]'] = (np.count_nonzero(np.logical_and(0.6 < vector, vector <= 0.7)) /
                                           vector.shape[0])
        mf['% of values in [0.7, 0.8]'] = (np.count_nonzero(np.logical_and(0.7 < vector, vector <= 0.8)) /
                                           vector.shape[0])
        mf['% of values in [0.8, 0.9]'] = (np.count_nonzero(np.logical_and(0.8 < vector, vector <= 0.9)) /
                                           vector.shape[0])
        mf['% of values in [0.9, 1]'] = np.count_nonzero(np.logical_and(0.9 < vector, vector <= 1)) / vector.shape[0]

        # Z-Score Distances in certain ranges of [0,1]
        mf['% of values with |Z-score| in [0,1)'] = np.count_nonzero(
            np.logical_and(0 <= zscore(vector), zscore(vector) < 1)) / vector.shape[0]
        mf['% of values with |Z-score| in [1,2)'] = np.count_nonzero(
            np.logical_and(1 <= zscore(vector), zscore(vector) < 2)) / vector.shape[0]
        mf['% of values with |Z-score| in [2,3)'] = np.count_nonzero(
            np.logical_and(2 <= zscore(vector), zscore(vector) < 3)) / vector.shape[0]
        mf['% of values with |Z-score| in [3, infinity)'] = np.count_nonzero(3 <= zscore(vector)) / vector.shape[0]

        return mf

    def ferrari_distance(self, df):
        """
        Calculates the meta-features from the distance-based vector, as per the Ferrari et al. 2015 paper.

        :param df: The dataset at hand.
        :type df: pd.DataFrame
        :return: A dictionary containing the meta-features.
        :rtype:  dict.
        """
        self.df = df
        if self.ferrari_distance_mf is None:
            self.distance_vector = distance_vector(self.df).reshape(-1, 1)
            self.ferrari_distance_mf = self._calculate_vector_properties(self.distance_vector)
            return self.ferrari_distance_mf
        else:
            return self.ferrari_distance_mf

    def correlation(self, df):
        """
        Calculates the meta-features from the distance-based vector, as per the Ferrari et al. 2015 paper.

        :param df: The dataset at hand.
        :type df: pd.DataFrame
        :return: A dictionary containing the meta-features.
        :rtype:  dict.
        """
        self.df = df
        rank_df = self.df.argsort(axis=0)
        if self.correlation_mf is None:
            self.correlation_vector = correlation_vector(rank_df).reshape(-1, 1)
            self.correlation_mf = self._calculate_vector_properties(self.correlation_vector)
            return self.correlation_mf
        else:
            return self.correlation_mf

    def cad(self, df):
        """
        Calculates the meta-features from the distance-based vector, as per the Ferrari et al. 2015 paper.

        :param df: The dataset at hand.
        :type df: pd.DataFrame
        :return: A dictionary containing the meta-features.
        :rtype:  dict.
        """
        self.df = df
        if self.correlation_vector is None:
            rank_df = self.df.argsort(axis=0)
            self.correlation_vector = correlation_vector(rank_df).reshape(-1, 1)
        if self.distance_vector is None:
            self.distance_vector = distance_vector(self.df).reshape(-1, 1)
        return self._calculate_vector_properties(MinMaxScaler().fit_transform(self.distance_vector +
                                                                              self.correlation_vector))
