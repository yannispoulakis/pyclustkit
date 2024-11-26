import pandas as pd
import numpy as np
from ast import literal_eval

from scipy.spatial.distance import pdist

x = pd.read_csv("tests/dev-single-cvi/preprocessed data.csv", header=None)
x = np.array(x)
print(x.shape)
y = pd.read_csv(r"../PyClust-Eval/tests/dev-single-cvi/AffinityPropagation.csv")
y = literal_eval(y.iloc[0]["Clustering Labels"])
y = np.array(y)
print(y.shape)

pdistances = pdist(x)

y = np.array(y)
cluster_membership = y[:, np.newaxis] == y[np.newaxis, :]

# upper triangular
m = cluster_membership.shape[0]
r, c = np.triu_indices(m, 1)
upper_cluster_membership = cluster_membership[r, c]

sorted_upper_dist_not_same = np.sort(pdistances[np.where(upper_cluster_membership == False)])[::-1]
sorted_upper_dist_same = np.sort(pdistances[np.where(upper_cluster_membership == True)])[::-1]

nb = len(sorted_upper_dist_not_same)
nw = len(sorted_upper_dist_same)

i,j = 0,0
s_plus = 0
s_minus = 0
while i < nw and j < nb:
    if j == nb-1:
        s_plus += (nw-(i+1)) * (nb-1)
        break

    if sorted_upper_dist_same[i] < sorted_upper_dist_not_same[j]:
        j += 1
        print("hula")
    else:
        print("hup")
        s_minus += nb - (j+1)  # All remaining i elements will be greater
        s_plus += j + 1
        i += 1
