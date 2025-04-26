import numpy as np
from scipy.spatial import distance_matrix


def fft(X, y , k) -> dict:
    """

    Args:
        X:
        y:
        k:

    Returns:
        (dict): A dictionary with keys the cluster labels and values the positions of the representatives in the
                original dataframe

    """
    assert k <= X.shape[0], f"Representative points {k} cannot be more than the dataset instances {X.shape[0]}."
    reps_full = {}
    for unique_cluster in np.unique(y):
        temp_df = X[np.where(y==unique_cluster)]
        cluster_center = temp_df.mean(axis=0).reshape(1,temp_df.shape[1])
        dist = distance_matrix(cluster_center, temp_df)
        reps=[np.argmax(dist)]

        for rep_number in range(0 ,k-1):
          dist = distance_matrix(temp_df[reps[rep_number]].reshape(1,temp_df.shape[1]), temp_df)
          argsorted_dist = dist.argsort()[0]
          i = -1
          while argsorted_dist[i] in reps:
            i += - 1
          reps.append(argsorted_dist[i])
        reps_full[str(unique_cluster)] = np.where(y == unique_cluster)[0][reps]
    return reps_full
