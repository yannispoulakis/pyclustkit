import numpy as np
from itertools import combinations


def dens_bw(X,y, sigma):
    """
    Calculates the inter-cluster variance based on number of points in the neighborhood of the middle point between
    clusters. See definition 1 in paper.

    :param clust_std:
    :type clust_std:
    :param X: Dataset records.
    :type X:  pd.DataFrame or np.array
    :param y: Cluster labels.
    :type y: pd.DataFrame or np.array
    :return: Inter-Cluster density.
    :rtype: float
    """
    clusters = list(set(y))
    cluster_centers = {i: X[y == i].mean(axis=0) for i in clusters}

    # Sum of density terms : (dens(midpoint(clust_a, clust_b) /max(dens(clust(a), dens(clust(b))
    dens_fraction_sum = 0
    for comb in combinations(clusters, r=2):
        centers_midpoint = (cluster_centers[comb[0]] + cluster_centers[comb[1]]) / 2
        distances = np.linalg.norm(X[np.logical_or(y==comb[0], y==comb[1])] - centers_midpoint, axis=1)
        midpoint_dens = np.sum(distances < sigma)

        gk_k = np.linalg.norm(X[np.logical_or(y==comb[0], y==comb[1])] - cluster_centers[comb[0]], axis=1)
        gkk_ = np.linalg.norm(X[np.logical_or(y==comb[0], y==comb[1])] - cluster_centers[comb[1]], axis=1)
        gk_k = np.sum(gk_k < sigma)
        gkk_ = np.sum(gkk_ < sigma)


        dens_fraction_sum += midpoint_dens / max([gk_k, gkk_])
    return dens_fraction_sum * (2/ (len(clusters) * (len(clusters)-1)))
