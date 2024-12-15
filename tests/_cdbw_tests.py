import matplotlib.pyplot as plt
from pyclustkit.eval.index_specifics.representatives import density, inter_density, cluster_separation, intra_dens, compactness
import numpy as np

# Density
d, mp, stdev, pts_in = density(blobs[0], blobs[1], rcr)

plt.scatter(blobs[0][:, 0], blobs[0][:, 1])
plt.scatter(blobs[0][:, 0][xxx["0"]], blobs[0][:, 1][xxx["0"]], c="red")
plt.scatter(blobs[0][:, 0][xxx["1"]], blobs[0][:, 1][xxx["1"]], c="yellow")
plt.scatter(mp[0], mp[1], c="green")
point_1 = blobs[0][:, 0][rcr[0][1][0][0]], blobs[0][:, 1][rcr[0][1][0][0]]
point_2 = blobs[0][:, 0][rcr[0][1][0][1]], blobs[0][:, 1][rcr[0][1][0][1]]
plt.plot((point_1[0], point_2[0]), (point_1[1], point_2[1]))
plt.scatter(blobs[0][:, 0][pts_in], blobs[0][:, 1][pts_in], c="black")
plt.show()

# Inter Density
idens = inter_density(d, len(np.unique(blobs[1])))

sep = cluster_separation(blobs[0], rcr, idens, 2)

intra_dens(blobs[0], blobs[1], xxx, 0.1)
a = compactness(s_range=[0.1, 0.2, 0.3], X=blobs[0], y=blobs[1], reps=xxx)