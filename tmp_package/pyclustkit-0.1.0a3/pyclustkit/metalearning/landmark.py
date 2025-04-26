"""
A script that contains the method for calculating landmark meta-features,
i.e. meta-features that derive from information provided after training a machine learning algorithm for the task.
"""
import numpy as np
from sklearn.cluster import MeanShift, DBSCAN, OPTICS
from ..eval import cvi
import logging

logging.basicConfig(level=logging.WARNING)

class CVIMF:
    def __init__(self):
        self.algorithms = {"meanshift": MeanShift().fit_predict, "dbscan": DBSCAN().fit_predict,
                           "optics": OPTICS().fit_predict}
        pass

    def calculate_cvi(self, x, cvi_name, algorithm):
        """

        Args:
            x (np.ndarray or pd.DataFrame): The data to calculate meta-features
            cvi_name (string): Name of the CVI to calculate
            algorithm (string): Name of the algorithm that partitions data

        Returns:
            float: The CVI value calculated
        """

        if hasattr(self, algorithm + "_labels"):
            pass
        else:
            setattr(self, algorithm + "_labels", self.algorithms[algorithm](x))

        labels = getattr(self, algorithm + "_labels")
        if len(np.unique(labels)) <= 1:
            logging.warning(f"Labels produced by {algorithm} equal to 1, score based "
                                                f"meta-feature {algorithm}-{cvi_name} will default to NaN")
            return None

        cvi_value = getattr(cvi, cvi_name)(x, labels)
        setattr(self, algorithm + f"_{cvi}", cvi_value)
        return cvi_value
