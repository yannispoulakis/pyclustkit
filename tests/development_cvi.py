# Generate synthetic data
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=15000, centers=4, cluster_std=0.60, random_state=0)
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)




# -------------------Calinski-Harabasz-----------------------------------------
from pyclustkit.eval import CVIToolbox
from sklearn.metrics import calinski_harabasz_score
cvit = CVIToolbox(X,y)
ch = cvit.calinski_harabasz()
ch_2 = calinski_harabasz_score(X, y)
print(ch, ch_2)

# -------------------------------------Silhouette----------------------------------------------------
from pyclustkit.eval.core import distances, intra_cluster_distances, inter_cluster_distances

d = distances(X)
id = intra_cluster_distances(d, y)
id = {x: np.mean(y, axis=1) for x, y in id.items()}
ied = inter_cluster_distances(d, y)


per_point_min_dist_of_nearest = {}
for label in set(y):
    relevant_keys = [key for key in ied.keys() if label in key]

    per_point_min_inter_dist = []
    for comb in relevant_keys:
        dist_df = ied[comb]
        if comb[0] == label:
            mean_arr = np.mean(dist_df, axis=1).reshape(-1, 1)
            print(mean_arr.shape)
        else:
            mean_arr = np.mean(dist_df, axis=0).reshape(-1,1)
        per_point_min_inter_dist.append(mean_arr)
    per_point_min_dist_of_nearest[label] = np.hstack(per_point_min_inter_dist)

per_point_min_dist_of_nearest[0].shape

sum = 0
counter = 0
for key in id:
    for row in range(0,id[key].shape[0]):
        counter +=1
        a_i = id[key][row]
        b_i = np.min(per_point_min_dist_of_nearest[key][row])
        s_i = (b_i -  a_i) / (np.max([b_i, a_i]))
        sum += s_i

print(sum/counter)

from sklearn.metrics import silhouette_score
sc = silhouette_score(X,y)
print(sc)

# Development/DaviesBouldin
from pyclustkit.eval.core import (pairwise_cluster_centers_distances, sum_distances_from_cluster_centers,
                                  cluster_centers, distances_from_cluster_centers)
from itertools import combinations
from sklearn.metrics import davies_bouldin_score
ccenters = cluster_centers(X, y)
ccenters_distances = pairwise_cluster_centers_distances(ccenters)

dfcc = distances_from_cluster_centers(X, y, ccenters)
sdfcc = sum_distances_from_cluster_centers(dfcc)
for key in sdfcc:
    sdfcc[key] = sdfcc[key] / len(y[y==key])

max_rij_sum = 0
for key in sdfcc:
    relevant_combs = [tup for tup in list(combinations(sdfcc.keys(), r=2)) if key in tup]
    print(relevant_combs)
    print("-----------------------------")
    ri_j = []
    for comb in relevant_combs:
        clust_dist = ccenters_distances[comb]
        si_plus_sj = sdfcc[comb[0]] + sdfcc[comb[1]]
        ri_j.append(si_plus_sj/clust_dist)
    max_rij_sum += max(ri_j)

db = max_rij_sum/ len(sdfcc.keys())
print(db)
print(davies_bouldin_score(X,y))
