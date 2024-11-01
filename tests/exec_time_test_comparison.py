"""
This test includes the necessary code to compare the PyClust execution time with the python Library ClusterFeatures and
R library: clusterCrit. clusterCrit has been tested in native R and clusterFeatures was modified to be compatible with
the latest updates of its components.
"""
import matplotlib.pyplot as plt
from pyclustkit.eval import CVIToolbox
from sklearn.datasets import make_blobs
import time
from ClustersFeatures import *
import json
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
n_samples_list = [1000, 5000, 10000, 20000]

# Compare with ClusterFeats
results = {"ClusterFeats": [], "PyClust": []}
for j in range(1, 10):
    exec_times_cluster_feat = []
    for i in n_samples_list:
        x, y = make_blobs(n_samples=i, n_features=3)
        x = pd.DataFrame(x)
        x['target'] = y
        start_time = time.time()
        CC = ClustersCharacteristics(pd.DataFrame(x), label_target="target")
        asd = CC.IndexCore_compute_every_index()
        exec_time = time.time() - start_time
        exec_times_cluster_feat.append(exec_time)
        results["ClusterFeats"].append(exec_times_cluster_feat)

    exec_times = []
    for i in n_samples_list:
        x, y = make_blobs(n_samples=i, n_features=3)
        cvit = CVIToolbox(x, y)
        start_time = time.time()
        cvit.calculate_icvi(exclude=["g_plus", "gamma", "tau", "cdbw", "ksq_detw"])
        exec_times.append(time.time() - start_time)
        results["PyClust"].append(exec_times)

with open(r"tests\pyclust_vs_clusterfeats.json", "w") as f:
    json.dump(results, f)

# Compare with Scikit-Learn
results = {"Scikit": [], "PyClust": []}
for j in range(1, 6):
    exec_times_scikit = []
    for i in n_samples_list:
        x, y = make_blobs(n_samples=i, n_features=3)
        x = pd.DataFrame(x)
        x['target'] = y
        start_time = time.time()
        sc = silhouette_score(x,y)
        db = davies_bouldin_score(x,y)
        ch = calinski_harabasz_score(x,y)
        exec_times_scikit.append(time.time() - start_time)
        results["Scikit"].append(exec_times_scikit)

    exec_times_pyclust = []
    for i in n_samples_list:
        x, y = make_blobs(n_samples=i, n_features=3)
        x = pd.DataFrame(x)
        x['target'] = y
        cvit = CVIToolbox(x, y)
        start_time = time.time()
        cvit.calculate_icvi(cvi=["silhouette", "davies_bouldin", "calinski_harabasz"])
        exec_times_pyclust.append(time.time() - start_time)
        results["PyClust"].append(exec_times_pyclust)


with open(r"tests\pyclust_vs_scikit.json", "w") as f:
    json.dump(results, f)

# Compare with R ClusterCrit
results = { "PyClust": []}
for j in range(1, 6):
    exec_times = []
    for i in n_samples_list:
        x, y = make_blobs(n_samples=i, n_features=3)
        cvit = CVIToolbox(x, y)
        start_time = time.time()
        cvit.calculate_icvi(exclude=["cdbw", "gdi61", "gdi62", "gdi63"])
        exec_times.append(time.time() - start_time)
        results["PyClust"].append(exec_times)

with open(r"tests\pyclust_vs_.json", "w") as f:
    json.dump(results, f)
"""
Below we visualize the results of the cvi calculations
exec_times_r = [0.09410405,  3.72401905, 18.70204592, 93.09587502]
exec_times = [0.2044992446899414, 4.025498628616333, 16.740500688552856, 69.4884991645813]
exec_times_cluster_feat = [18.619999647140503, 40.886000633239746, 89.54250168800354, 305.6704993247986]
"""

# Example data: lists of execution times
iterations = n_samples_list  # The number of iterations or budget

# Create the plot
plt.figure(figsize=(10, 6))

plt.plot(iterations, exec_times_cluster_feat, marker='o', linestyle='-', color='blue', label='ClusterFeatures')
plt.plot(iterations, exec_times_r, marker='o', linestyle='-', color='red', label='ClusterCrit')

plt.plot(iterations, exec_times, marker='o', linestyle='-', color='green', label='PyClust')

# Adding titles and labels
plt.title('Comparison of CVI Calculation Times Between Frameworks')
plt.xlabel('No_Samples')
plt.ylabel('Execution Time (seconds)')
plt.legend()

# Show grid for better readability
plt.grid(True)

# Show the plot
plt.savefig('exec_times_comparison.png')