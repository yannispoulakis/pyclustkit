import numpy as np
from scipy.spatial.distance import pdist
from typing import Union, Tuple
from pyclustkit.eval.core._utils import upper_triu

def upper_triangle(matrix):
    """Returns the upper triangle of values of symmetric matrix. Useful for collecting unique values in a pairwise
    distance matrix"""
    rows, cols = np.triu_indices(len(matrix), k=1)  # Get the indices of upper triangle excluding the diagonal
    return matrix[rows, cols]


def return_s(X, labels,
             precomputed_distances=False) -> Tuple[int, int, int, int]:
    """
    Calculates the necessary values for the gamma and tau cluster validity indices. Returns the following:
    s_plus : Times points within the same cluster have distance less than points in different clusters.
    s_minus : Times points within the same cluster have distance more than points in different clusters.
    nb : Distances between all the pairs of points that are in different clusters
    nw : Distances between all the pairs of points that are in the same cluster.
    """
    if precomputed_distances is False:
        pairwise_distances = pdist(X)

    else:
        assert type(precomputed_distances) is np.ndarray, "Provided distance matrix should be np.ndarray"
        assert precomputed_distances.shape[0] == X.shape[0], ("Provided distance matrix should be of shape "
                                                              "(X.shape[0],X.shape[1])")

        pairwise_distances = upper_triu(precomputed_distances)

    cluster_membership = np.zeros((len(X), len(X)), dtype=bool)

    # Iterate over each pair of points to define cluster membership
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            if labels[i] == labels[j]:
                cluster_membership[i, j] = True
                cluster_membership[j, i] = True

    # sort distances to compare progressively points and enhance efficiency.
    upper_cluster_membership = upper_triangle(cluster_membership)
    sorted_upper_dist_not_same = -np.sort(-pairwise_distances[np.where(upper_cluster_membership == False)])
    sorted_upper_dist_same = -np.sort(-pairwise_distances[np.where(upper_cluster_membership == True)])

    # used for the denominator in tau index
    nb = len(sorted_upper_dist_not_same)
    nw = len(sorted_upper_dist_same)

    # calculate differences
    j = 0
    s_plus = 0
    s_minus = 0
    inner_break = False
    for idx, val_true in enumerate(sorted_upper_dist_same):
        if inner_break:
            break
        for i in range(j, len(sorted_upper_dist_not_same)):
            if val_true < sorted_upper_dist_not_same[i]:
                s_plus += 1
                # if nn[-1] > nw[i] loop terminates, i * 1 is added to s_plus
                if j == len(sorted_upper_dist_not_same) - 1:
                    s_plus += 1 * (len(sorted_upper_dist_same) - idx)
                    inner_break = True
                continue
            else:
                s_minus += len(sorted_upper_dist_not_same) - i - 2
                j = i
                break

    return s_plus, s_minus, nb, nw
