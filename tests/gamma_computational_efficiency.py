from sklearn.datasets import make_blobs
import time
import numpy as np
from scipy.spatial.distance import pdist

x, labels = make_blobs(n_samples=10000, n_features=2)
labels = np.array(labels)


start_time = time.time()
cluster_membership = labels[:, np.newaxis] == labels[np.newaxis,:]
print(time.time() - start_time)
upper_indices = np.triu_indices(cluster_membership.shape[0], k=1)
cm_upper = cluster_membership[upper_indices]

dists = pdist(x)

dist_in_same = np.sort(dists[np.where(cm_upper==True)])
dist_not_in_same = np.sort(dists[np.where(cm_upper==False)])
print(len(dist_in_same) + len(dist_not_in_same))

s_plus = 0
s_minus = 0
j = 0
for idx, val_true in enumerate(dist_in_same):
    print(j)
    for i in range(j, len(dist_not_in_same)):
        if val_true < dist_not_in_same[i]:
            s_plus += 1
        else:
            s_minus += len(dist_not_in_same) - i
            j= i
            print("i break")
            break




j = 0
for i in [1, 2]:
    for x in range(j, 10):
        print(x)
        if x == 5:
            j =9
            break
