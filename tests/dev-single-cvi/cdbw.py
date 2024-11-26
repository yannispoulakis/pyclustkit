

import pandas as pd
from pyclustkit.eval.index_specifics.representatives import *
from pyclustkit.eval.index_specifics.FFT import *
from ast import literal_eval
import numpy as np

x = pd.read_csv("tests/dev-single-cvi/preprocessed data.csv", header=None)
x = np.array(x)
print(x.shape)

y = pd.read_csv(r"../PyClust-Eval/tests/dev-single-cvi/AffinityPropagation.csv")
y = literal_eval(y.iloc[0]["Clustering Labels"])
y = np.array(y)

# (a) Find Representatives
reps = fft(x,y, k=5)

# cohesion and compactness

# x, cluster_centers, reps
from pyclustkit.eval.core._common_processes import cluster_centers
cc = cluster_centers(x, y)

total_cardinality = 0
total_stdev = 0
for cluster in reps.keys():
    x_temp = x[np.where(y==int(cluster))]
    radius_for_cardinality = np.std(x_temp)

    no_reps_per_cluster = len(reps[list(reps.keys())[0]])

    for cluster_rep in reps[cluster]:

        # (A) Shrink representative.
        shrunk_representative = x[cluster_rep] + 0.5 * (cc[0] - x[0])

        # (B) Calculate Distances between shrunk representative and other points in the cluster.
        dist_matrix = distance_matrix(np.reshape(shrunk_representative, (1, x.shape[1])), x_temp)

        rep_cardinality = (dist_matrix <= radius_for_cardinality).sum() / x_temp.shape[0]

        total_cardinality += rep_cardinality

    total_stdev += radius_for_cardinality

dens_cl = total_cardinality / no_reps_per_cluster
intra_dens_  = dens_cl / (total_stdev/len(reps.keys()))




coh, comp = cohesion(x,y,reps)
