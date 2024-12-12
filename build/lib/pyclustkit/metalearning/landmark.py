"""
A script that contains the method for calculating landmark meta-features,
i.e. meta-features that derive from information provided after training a machine learning algorithm for the task.
"""

from sklearn.cluster import MeanShift, DBSCAN, OPTICS
from ..eval import cvi


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

        cvi_value = getattr(cvi, cvi_name)(x, getattr(self, algorithm + "_labels"))
        setattr(self, algorithm + f"_{cvi}", cvi_value)
        return cvi_value
