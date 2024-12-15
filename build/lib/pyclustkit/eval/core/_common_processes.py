from scipy.spatial.distance import cdist
from scipy.spatial.distance import directed_hausdorff
from itertools import combinations
from pyclustkit.eval.core._utils import sum_of_upper_triu
import numpy as np
from scipy.spatial.distance import pdist
from typing import Union, Tuple
from pyclustkit.eval.core._utils import upper_triu

#TODO: Check which of np.linalg.norm or cdist or pdist is faster (cdist< norm)


def distances(x):
    return cdist(x, x)


def sum_distances(distances):
    m = distances.shape[0]
    r, c = np.triu_indices(m, 1)
    return np.sum(distances[r, c])


# Data Center operations
def data_center(x):
    return np.mean(x, axis=0).reshape(1, x.shape[1])


def distances_from_data_center(x, dcenter):
    return np.linalg.norm(x - dcenter, axis=1)


# ------ Cluster Centers operations ------
def cluster_centers(x, clabels):
    """

    Args:
        x (np.ndarray): The dataset instances.
        clabels (np.ndarray): The cluster labels.

    Returns:
        dict: {cluster_label: cluster_center} -> {int: np.ndarray}
    """
    return {i: x[clabels == i].mean(axis=0) for i in np.unique(clabels)}


def pairwise_cluster_centers_distances(ccenters, distance_metric='euclidean'):
    """

    Args:
        ccenters (dict): The cluster centers derived from function: cluster_centers()
        distance_metric (str): Any metric supported by scipy.spatial.distance.cdist()

    Returns:
        dict: {(cluster_label_1, cluster_label_2): dist(cluster_center_1, cluster_center_2)} -> {tuple(int,int): float}
    """
    return {(i, j): cdist(ccenters[i].reshape(1, -1), ccenters[j].reshape(1, -1),
                          metric=distance_metric)[0][0] for i, j in combinations(ccenters.keys(), r=2)}


def distances_from_cluster_centers(x, clabels, ccenters):
    """

    Args:
        x (np.ndarray): The dataset instances
        clabels (np.ndarray): The cluster labels
        ccenters (dict): The cluster centers in a dictionary of {cluster_label: cluster_center}

    Returns:
        dict: in the format of {cluster_label: distances_from_cluster_center}

    """
    return {i: cdist(x[clabels == i], ccenters[i].reshape(1, -1)) for i in np.unique(clabels)}


def sum_distances_from_cluster_centers(dfcc):
    """

    Args:
        x (np.ndarray): The dataset instances
        clabels (np.ndarray): The cluster labels
        ccenters (dict): The cluster centers in a dictionary of {cluster_label: cluster_center}

    Returns:
        dict: in the format of {cluster_label: distances_from_cluster_center}

    """
    return {i: j.sum() for i, j in dfcc.items()}


def pairwise_sum_distances_from_cluster_centers(sum_distances_from_centers):
    return {(i, j): sum([sum_distances_from_centers[i], sum_distances_from_centers[j]]) for i, j in
            combinations(list(sum_distances_from_centers.keys()), r=2)}


def distances_from_other_cluster_centers(x, y, ccenters):
    ccenters_matrix = np.array(list(ccenters.values()))
    for key in ccenters:
        ccenters[key] = cdist(x[y == key], np.vstack((ccenters_matrix[:key], ccenters_matrix[key + 1:])))
    return ccenters


# Dispersion & Scatter matrices
def between_group_scatter_matrix(X, cluster_labels) -> np.array:
    # Get the number of clusters
    unique_clusters = np.unique(cluster_labels)
    k = len(unique_clusters)

    # Calculate the overall mean of the dataset
    overall_mean = np.mean(X, axis=0)

    # Initialize the between-group scatter matrix to zeros
    n_features = X.shape[1]
    S_B = np.zeros((n_features, n_features))

    # Loop through each cluster
    for cluster in unique_clusters:
        # Get the data points in this cluster
        cluster_points = X[cluster_labels == cluster]

        # Calculate the mean of the cluster
        cluster_mean = np.mean(cluster_points, axis=0)

        # Get the number of data points in this cluster
        n_cluster_points = cluster_points.shape[0]

        # Calculate the difference between the cluster mean and the overall mean
        mean_diff = (cluster_mean - overall_mean).reshape(-1, 1)
        # Add the contribution to the between-group scatter matrix
        S_B += n_cluster_points * (mean_diff @ mean_diff.T)

    return S_B


def total_group_scatter(X, cluster_labels) -> np.array:
    """

    :param x: The dataset
    :type x: np.array
    :param y: The cluster labels
    :param ccenters_from_dcenter: Difference of cluster centers to the data center
    :return:
    """
    # Get the number of clusters
    unique_clusters = np.unique(cluster_labels)
    k = len(unique_clusters)

    # Calculate the overall mean of the dataset
    overall_mean = np.mean(X, axis=0)

    # Initialize the between-group scatter matrix to zeros
    n_features = X.shape[1]
    S_B = np.zeros((n_features, n_features))

    # Loop through each cluster
    for cluster in unique_clusters:
        # Get the data points in this cluster
        cluster_points = X[cluster_labels == cluster]

        # Calculate the mean of the cluster
        cluster_mean = np.mean(cluster_points, axis=0)

        # Get the number of data points in this cluster
        n_cluster_points = cluster_points.shape[0]

        # Calculate the difference between the cluster mean and the overall mean
        mean_diff = (cluster_mean - overall_mean).reshape(-1, 1)
        # Add the contribution to the between-group scatter matrix
        S_B += n_cluster_points * (mean_diff @ mean_diff.T)


def within_group_scatter_matrices(x, clabels, ccenters):
    dif = {i: x[clabels == i] - ccenters[i] for i in ccenters}
    scatter_matrices = {i: j.T @ j for i, j in dif.items()}
    return scatter_matrices


def total_within_group_scatter_matrix(x, wg_scatter_matrices):
    TWG = np.zeros((x.shape[1], x.shape[1]))
    for scatter_matrix in wg_scatter_matrices.values():
        TWG += scatter_matrix
    return TWG


def total_scatter_matrix(X):
    """
    Calculate the total scatter matrix S_T for the dataset X.

    Parameters:
    - X: A numpy array of shape (n_samples, n_features), representing the dataset.

    Returns:
    - S_T: The total scatter matrix.
    """
    # Calculate the global mean of the dataset
    global_mean = np.mean(X, axis=0)

    # Initialize the total scatter matrix to zeros
    n_features = X.shape[1]
    S_T = np.zeros((n_features, n_features))

    # Loop through each data point
    for x in X:
        # Calculate the difference between the data point and the global mean
        mean_diff = (x - global_mean).reshape(-1, 1)

        # Add the contribution to the total scatter matrix
        S_T += mean_diff @ mean_diff.T

    return S_T


# inter-intra
def intra_cluster_distances(pdistances, clabels):
    return {i: pdistances[np.ix_(np.where(clabels == i)[0], np.where(clabels == i)[0])] for i
            in np.unique(clabels)}


def sum_intra_cluster_distances(intra_cdistances):
    return {i: sum_of_upper_triu(intra_cdistances[i]) for i, j in intra_cdistances.items()}


def inter_cluster_distances(pdistances, clabels):
    return {(i, j): pdistances[np.ix_(np.where(clabels == i)[0], np.where(clabels == j)[0])] for i, j in
            combinations(np.unique(clabels), r=2)}


def sum_inter_cluster_distances(inter_cdistances):
    return {i: inter_cdistances[i].sum() for i, j in inter_cdistances.items()}


def cluster_centers_from_data_center(ccenters, dcenter):
    return {i: cdist(j, dcenter) for i, j in ccenters.items()}


def max_cdistances(inter_cdistances: dict):
    """

    :param inter_cdistances:
    :type inter_cdistances:
    :return:
    :rtype:
    """
    return {i: np.max(j) for i, j in inter_cdistances.items()}


def min_cdistances(inter_cdistances: dict):
    """

    :param inter_cdistances:
    :type inter_cdistances:
    :return:
    :rtype:
    """
    return {i: np.min(j) for i, j in inter_cdistances.items()}


def pairwise_hausdorff(x, y):
    return {(i, j): max(directed_hausdorff(x[y == i], x[y == j])[0], directed_hausdorff(x[y == j], x[y == i])[0]) for
            i, j in combinations(np.unique(y), r=2)}


def trace(matrix):
    return np.trace(matrix)

    return S_B


# S
def return_s(X, labels,
             precomputed_distances=False) -> Tuple[int, int, int, int]:
    """
    Calculates the necessary values for the gamma and tau cluster validity indices. Returns the following:
    s_plus : Times that points within the same cluster have distance less than points in different clusters.
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

    labels = np.array(labels)
    cluster_membership = labels[:, np.newaxis] == labels[np.newaxis,:]

    # sort distances to compare progressively points and enhance efficiency.
    upper_cluster_membership = upper_triu(cluster_membership)
    sorted_upper_dist_not_same = -np.sort(-pairwise_distances[np.where(upper_cluster_membership == False)])
    sorted_upper_dist_same = -np.sort(-pairwise_distances[np.where(upper_cluster_membership == True)])

    # used for the denominator in tau index
    nb = len(sorted_upper_dist_not_same)
    nw = len(sorted_upper_dist_same)

    # calculate differences
    i,j = 0,0
    s_plus = 0
    s_minus = 0
    while i < nw and j < nb:
        if j == nb - 1:
            s_plus += (nw - (i + 1)) * (nb - 1)
            break
        if sorted_upper_dist_same[i] < sorted_upper_dist_not_same[j]:  # All remaining j elements will be greater
            j += 1

        else:
            s_minus += nb - j # All remaining i elements will be greater
            s_plus += j
            i += 1

    return s_plus, s_minus, nb, nw



