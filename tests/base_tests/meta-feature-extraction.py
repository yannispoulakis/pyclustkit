"""
Test for the meta-feature extraction functionalities based on the MFExtractor class.
"""

from pyclustkit.metalearning import MFExtractor
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs, make_moons

# Test numeric dataset Iris
df = pd.read_csv(r"tests/iris.csv")

mfe = MFExtractor(data=np.array(df))
mfe.meta_features
mfe.calculate_mf(included_in="AutoClust")

# Meta-Feature results can be retrieved in various ways.
# (a) search_type can be any of [names, values, full_search]
print(mfe.search_mf(search_type="names"))
print(mfe.search_mf(search_type="values"))
print(mfe.search_mf(search_type="full_search"))



x, y = make_blobs(n_samples=100, n_features=2)

from pyclustkit.metalearning.landmark import CVIMF
cvimf = CVIMF()
cvimf.calculate_cvi(x)
cvimf.optics_labels