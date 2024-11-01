import numpy as np
from sklearn.datasets import make_blobs
from pyclustkit.eval.core import cluster_centers, within_group_scatter_matrices
import math
from itertools import combinations
from pyclustkit.eval.core import find_midpoint
from scipy.spatial.distance import cdist

x, y = make_blobs(n_samples=100)

# sigma : ok!
clusters = np.unique(y)
cstd = {}
for clust in clusters:
    cstd[clust] = np.var(x[y == clust], axis=0)
sigma = sum([np.linalg.norm(x) for x in cstd.values()])
print(sigma)
sigma = math.sqrt(sigma) / len(clusters)
print(f'sigma {sigma}')

ccenters = cluster_centers(x, y)
wgm = within_group_scatter_matrices(x, y, ccenters)
wgm = {i: j / len(x[y == i]) for i, j in wgm.items()}
wgm = {i: np.diag(j) for i, j in wgm.items()}
sigma = sum([np.linalg.norm(x) for x in wgm.values()])
print(sigma)
sigma = (1 / len(np.unique(y))) * math.sqrt(sigma)
print(f'sigma {sigma}')

# HKK
midpoints = {(i, j): find_midpoint(ccenters[i].reshape(1, -1), ccenters[j].reshape(1, -1)) for i, j in
             combinations(ccenters.keys(), r=2)}

midpoint_distances = {i: cdist(midpoints[i], np.concatenate([x[y == i[0]], x[y == i[1]]]))
                      for i, j in midpoints.items()}
Hkk = {i: np.sum(j < sigma) for i, j in midpoint_distances.items()}

midpoints_2 = []
midpoint_distances_2 = {}
for comb in combinations(clusters, r=2):
    centers_midpoint = (ccenters[comb[0]] + ccenters[comb[1]]) / 2
    print(centers_midpoint.shape)
    midpoints_2.append(centers_midpoint)
    midpoint_distances_2[comb] = np.linalg.norm(x[np.logical_or(y == comb[0], y == comb[1])] - centers_midpoint, axis=1)




np.sort(midpoint_distances[(0,1)]) == np.sort(midpoint_distances_2[(0,1)])


for comb in midpoint_distances:
    print(comb, np.sum(midpoint_distances[comb] < sigma))


for comb in midpoint_distances_2:
    print(comb, np.sum(midpoint_distances_2[comb] < sigma))
