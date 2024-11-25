from pyclustkit.metalearning import MFExtractor
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs, make_moons

df = pd.read_csv(r"D:\PyClust-Eval\tests\iris.csv")

mfe = MFExtractor(data=np.array(df))
mfe.calculate_mf()


x, y = make_blobs(n_samples=100, n_features=2)
mfe = MFExtractor(data=x)
mfe.calculate_mf()
mfe.calculate_mf(category="landmark")
mfe.search_mf(search_type='values')

# Retrieve all meta-features of Souto
mfe.search_mf(included_in="AutoClust", search_type="names")
x, y = make_blobs(n_samples=100, n_features=2)

from pyclustkit.metalearning.landmark import CVIMF
cvimf = CVIMF()
cvimf.calculate_cvi(x)
cvimf.optics_labels