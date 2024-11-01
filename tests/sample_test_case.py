"""
Suite of basic tests:
    (a) All CVI return a value
    (b) Can include and exclude subsets successfully from the calculate_icvi function
"""

from pyclustkit.eval import CVIToolbox
import pandas as  pd
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv("tests/preprocessed data.csv")
df = np.array(df)
km = KMeans(n_clusters=4)
labels = km.fit_predict(df)

# Test 1 - All CVI return a value/ No CVI
cvit = CVIToolbox(df, labels)
cvit.calculate_icvi()

print(f"All CVI results return numeric values: {all(isinstance(v, (int, float)) for v in cvit.cvi_results.values())}")
print(f"Results contain measurement for {len(cvit.cvi_results.keys())} CVI")

# Test 2 - Custom index selection
cvit = CVIToolbox(df, labels)
cvit.calculate_icvi(cvi=["silhouette", "ray_turi", "gamma"])

# Test 3 - Index exclusion
cvit = CVIToolbox(df, labels)
cvit.calculate_icvi(exclude=["silhouette", "ray_turi", "gamma"])
print(len(cvit.cvi_results.keys()))

# Test 4 - Errors
cvit = CVIToolbox(df, labels)
cvit.calculate_icvi(exclude=["siette", "ray_i", "gamma"])
cvit.calculate_icvi(cvi=["siette", "ray_i", "gamma"])
cvit.calculate_icvi(cvi=["silhouette", "ray_turi", "gamma"], exclude=["trace_wib", "gamma"])
print(len(cvit.cvi_results.keys()))