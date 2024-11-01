from scipy.spatial import distance_matrix, distance
from scipy.spatial.distance import cdist
import itertools
import numpy as np
import math


def closest_representatives(X: np.array, reps_positions: object) -> object:
    """For each representative of each cluster find the closest representative of another cluster."""
    X = np.array(X)
    cluster_combinations = list(itertools.combinations(reps_positions.keys(), 2))
    closest_reps_master = dict([(x, {}) for x in reps_positions.keys()])

    for comb in cluster_combinations:
        cluster_0_reps = X[reps_positions[comb[0]]]
        cluster_1_reps = X[reps_positions[comb[1]]]
        dist_matrix = cdist(cluster_0_reps, cluster_1_reps)

        # Find the closest representatives according to dist matrix
        closest_reps_cluster_zero = np.argmin(dist_matrix, axis=0)
        closest_reps_cluster_one = np.argmin(dist_matrix, axis=1)

        # Correlate indices from dist matrix to true indices from the original array
        true_closest_reps = reps_positions[comb[1]][closest_reps_cluster_one]
        closest_reps_master[comb[0]][str(comb[1])] = true_closest_reps
        true_closest_reps = reps_positions[comb[0]][closest_reps_cluster_zero]
        closest_reps_master[comb[1]][str(comb[0])] = true_closest_reps

    return closest_reps_master


def respective_closest_representatives(reps_positions, closest_reps):
    """Finds the respective closest representatives per cluster combination. That is pairs of representatives from
    different clusters that have each other as closest representative."""

    cluster_combinations = list(itertools.combinations(reps_positions.keys(), 2))
    rcr = []
    for comb in cluster_combinations:
        reps_1 = reps_positions[comb[0]]
        reps_2 = reps_positions[comb[1]]
        creps_1 = closest_reps[comb[0]][comb[1]]
        creps_2 = closest_reps[comb[1]][comb[0]]
        rcr_per_cluster_comb = []

        for i in range(0, len(reps_1)):
            rep_check_idx = np.where(reps_2 == creps_1[i])
            if creps_2[rep_check_idx] == reps_1[i]:
                rcr_per_cluster_comb.append((reps_1[i], creps_1[i]))
        rcr.append((comb, rcr_per_cluster_comb))
    return rcr

def midpoint(X, idx_0, idx_1):
    midpoint_coords = []
    for i in range(0, X.shape[1]):
        midpoint_coords.append((X[idx_0, i] + X[idx_1, i] )/ 2)
    return np.array(midpoint_coords)


def density(X, y, rcr):
    X = np.array(X)
    y = np.array(y)
    densities = {}
    for comb in rcr:
      densities.setdefault(comb[0][0], {})
      densities.setdefault(comb[0][1], {})
      # keep relevant clusters
      cluster_i = X[y == int(comb[0][0])]
      cluster_j = X[y == int(comb[0][1])]
      stdev_i = np.std(cluster_i)
      stdev_j = np.std(cluster_j)
      avg_stdev = (stdev_i + stdev_j) / 2

      sum = 0
      for crep_pair in comb[1]:
          crep_distance = distance.euclidean(X[crep_pair[0]], X[crep_pair[1]])

          # cardinality
          middle_pt = midpoint(X, crep_pair[0], crep_pair[1])
          distances_from_center = distance_matrix(np.reshape(middle_pt, (1, X.shape[1])), X)
          in_radius = np.where(distances_from_center <= avg_stdev)[1]
          cardinality_denominator = (cluster_j.shape[0] + cluster_i.shape[0])
          cardinality = in_radius.shape[0] / cardinality_denominator

          sum += (crep_distance / (2 * avg_stdev)) * cardinality

      density_ = sum * (1/len(rcr))
      densities[comb[0][0]][comb[0][1]] = density_
      densities[comb[0][1]][comb[0][0]] = density_
    return densities

def inter_density(densities):
    max_dens_sum = 0
    for key in densities.keys():
        max_dens_sum += max(densities[key].values())
    return max_dens_sum / len(densities.keys())


def cluster_separation(X, rcr, inter_cluster_density):
    inter_cluster_distances = {}
    for comb in rcr:
        rcr_distances = []
        inter_cluster_distances.setdefault(comb[0][0], {})
        inter_cluster_distances.setdefault(comb[0][1], {})
        for pair in comb[1]:
            rcr_dist = math.dist(X[pair[0]], X[pair[1]])
            rcr_distances.append(rcr_dist)
        inter_cluster_distances[comb[0][0]][comb[0][1]] = np.mean(rcr_distances)
        inter_cluster_distances[comb[0][1]][comb[0][0]] = np.mean(rcr_distances)

    sum_ = 0
    for key in inter_cluster_distances.keys():
        sum_ += min(inter_cluster_distances[key].values())

    sum_ = sum_ / len(inter_cluster_distances.keys())
    return sum_ / (1 + inter_cluster_density)


def intra_dens(X, y, reps, s):
    """
    Computes the relative intra-cluster density. That is the number of points within range of the
    representative points of each cluster.
    paper: Definition 6

    :param X: Dataset to calculate intra dens.
    :type X: np.array
    :param np.array y: Cluster labels for the dataset.
    :param dic reps: Dictionary holding the per-cluster representatives. Index position based on original array.
    :param s: A shrink factor for the representatives.
    :type s: float
    """

    total_card = []
    total_stdev = []
    for cluster in reps.keys():
        cluster_card = []
        X_temp = X[np.where(y==int(cluster))]
        radius = np.std(X_temp)
        total_stdev.append(radius)
        for cluster_rep in reps[cluster]:
            shrunk_rep = X[cluster_rep] - s
            dist_matrix = distance_matrix(np.reshape(shrunk_rep, (1,X.shape[1])), X_temp)
            rep_card = (dist_matrix <= radius).sum()
            cluster_card.append(rep_card/ X_temp.shape[0])
        total_card.append(sum(cluster_card))
    dens_cl = sum(total_card) /len(reps[list(reps.keys())[0]])
    return dens_cl / (len(np.unique(y) * sum(total_stdev)))


def cohesion(X,y,reps, s_range=np.arange(0.1, 0.9, step=0.1)):
    """
    returns cohesion of clusters. Definition 9 in paper.

    :param X: The dataset of concern.
    :type X: np.array or pd.DataFrame
    :param y: The cluster labels for the dataset.
    :type y: np.array or pd.DataFrame
    :param reps: The representatives of each cluster
    :type reps: dict
    :param s_range: A list of shrink factors for the representatives.
    :type s_range: list
    :return: Cohesion, compactness
    :rtype: float, float
    """
    total = []
    for i  in s_range:
        idens = intra_dens(X,y, reps,i)
        total.append(idens)
    compactness = sum(total) / len(s_range)
    intra_cluster_change = sum(np.ediff1d(total)) / (len(total) - 1)
    return compactness / (1 + intra_cluster_change), compactness

