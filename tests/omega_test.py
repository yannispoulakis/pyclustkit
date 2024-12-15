from pyclustkit.eval import CVIToolbox
import numpy as np

# Parameters for dataset
n_samples = 121  # Total number of samples
n_features = 2    # Number of features per sample
n_clusters = 3    # Number of clusters

# Generate random data points
X = np.random.rand(n_samples, n_features)

# Generate random cluster labels (0 to n_clusters - 1)
y = np.random.randint(0, n_clusters, n_samples)

ct = CVIToolbox(X,y)
ct.calculate_icvi()
ct.cvi_results

# <-------------------------Meta Learning------------------------------------------------------------->
from pyclustkit.metalearning import MFExtractor
mfe = MFExtractor(data=X)

mfe.calculate_mf()
mfe.meta_features