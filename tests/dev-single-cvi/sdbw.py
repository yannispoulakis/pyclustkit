import pandas as pd
import numpy as np
from ast import literal_eval
from pyclustkit.eval.core._common_processes import (within_group_scatter_matrices, cluster_centers,
                                                    distances_from_cluster_centers, distances_from_other_cluster_centers)
from pyclustkit.eval.core._utils import find_midpoint
from scipy.spatial.distance import pdist, cdist
import math
from itertools import combinations

x = pd.read_csv("tests/dev-single-cvi/preprocessed data.csv", header=None)
x = np.array(x)
print(x.shape)
y = pd.read_csv(r"../PyClust-Eval/tests/dev-single-cvi/AffinityPropagation.csv")
y = literal_eval(y.iloc[0]["Clustering Labels"])
y = np.array(y)
print(y.shape)


cc = cluster_centers(x, y)
wg_scatter_matrices = within_group_scatter_matrices(x, y,cc )
wg_scatter_matrices = {i: j / len(x[y == i]) for i, j in wg_scatter_matrices.items()}


# avg stdev of clusters / should be equal to sigma?
cluster_stds = []
for c in np.unique(y):
    std = np.std(x[y==c], axis=0)  # Standard deviation per dimension
    print(std)
    cluster_stds.append(np.linalg.norm(std))

sigma = np.mean(cluster_stds)


cluster_stds = []
for c in np.unique(y):
    std = np.var(x[y==c], axis=0)  # Standard deviation per dimension
    print(std)
    cluster_stds.append(np.linalg.norm(std))

sigma = np.mean(cluster_stds)
sigma = math.sqrt(sigma)

# avg stdev of clusters / should be equal to sigma?
cluster_stds = []
for c in np.unique(y):
    std = np.std(x[y==c])  # Standard deviation per dimension
    print(std)
    cluster_stds.append(std)

sigma = np.mean(cluster_stds)



# calculate sigma
sigma_2 = sum([np.linalg.norm(np.diag(x)) for x in wg_scatter_matrices.values()])
sigma_2 = (1 / len(np.unique(y))) * math.sqrt(sigma)




midpoints = {(i, j): find_midpoint(cc[i].reshape(1, -1), cc[j].reshape(1, -1)) for i, j in
             combinations(cc.keys(), r=2)}
midpoint_distances = {i: cdist(midpoints[i], np.concatenate([x[y == i[0]], x[y == i[1]]]))
                      for i, j in midpoints.items()}

Hkk = {i: np.sum(j < sigma) for i, j in midpoint_distances.items()}

gk = distances_from_cluster_centers(x, y, cc)
gk_ = distances_from_other_cluster_centers(x,y,cc)

gk[0]
sum_ = 0
for comb in combinations(cc.keys(), r=2):
    print(comb)
    k = comb[0]
    k_ = comb[1]
    if k_ > k:
        k_ += -1
    print(gk[k] < sigma)
    print(gk_[k][:, k_] < sigma)
    gkk_ = np.sum(gk[k] < sigma) + np.sum(gk_[k][:, k_] < sigma)

    k = comb[0]
    k_ = comb[1]
    if k_ < k:
        k += -1
    gk_k = np.sum(gk[k_] < sigma) + np.sum(gk_[k_][:, k] < sigma)

    sum_ += Hkk[comb] / max(gk_k, gkk_)












self.calculate_icvi(['sd_scat'])
S = self.cvi_results['sd_scat']




sum_ = 0
for comb in combinations(ccenters.keys(), r=2):
    k = comb[0]
    k_ = comb[1]
    if k_ > k:
        k_ += -1

    gkk_ = np.sum(gk[k] < sigma) + np.sum(gk_[k][:, k_] < sigma)

    k = comb[0]
    k_ = comb[1]
    if k_ < k:
        k += -1
    gk_k = np.sum(gk[k_] < sigma) + np.sum(gk_[k_][:, k] < sigma)

    sum_ += Hkk[comb] / max(gk_k, gkk_)

return (sum_ * (2 / (self.noclusters * (self.noclusters - 1)))) + S