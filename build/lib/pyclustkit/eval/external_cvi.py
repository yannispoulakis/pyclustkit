from sklearn.metrics import adjusted_rand_score, homogeneity_score, adjusted_mutual_info_score
from sklearn.metrics import completeness_score, fowlkes_mallows_score, homogeneity_completeness_v_measure
from sklearn.metrics import mutual_info_score, v_measure_score
import numpy as np

def compute_external_cvis(true_labels, cluster_labels):
    cvis = {}
    try:
        cvis['ARI'] = adjusted_rand_score(true_labels, cluster_labels)
    except Exception as e:
        print(e)
        cvis['ARI'] = np.nan
    try:
        cvis['Homogeneity'] = homogeneity_score(true_labels, cluster_labels)
    except:
        cvis['Homogeneity'] = np.nan
    try:
        cvis['AMI'] = adjusted_mutual_info_score(true_labels, cluster_labels)
    except:
        cvis['AMI'] = np.nan
    try:
        cvis['Completeness'] = completeness_score(true_labels, cluster_labels)
    except:
        cvis['Completeness'] = np.nan
    try:
        cvis['Fowlkes-Mallows Score'] = fowlkes_mallows_score(true_labels, cluster_labels)
    except:
        cvis['Fowlkes-Mallows Score'] = np.nan
    try:
        cvis['Homogeneity-Completeness V-Measure'] = homogeneity_completeness_v_measure(true_labels, cluster_labels)
    except:
        cvis['Homogeneity-Completeness V-Measure'] = np.nan
    try:
        cvis['Mutual Information'] = mutual_info_score(true_labels, cluster_labels)
    except:
        cvis['Mutual Information'] = np.nan
    try:
        cvis['V-Measure'] = v_measure_score(true_labels, cluster_labels)
    except:
        cvis['V-Measure'] = np.nan
    # cvis = pd.DataFrame.from_dict(cvis, orient='index').T
    return cvis