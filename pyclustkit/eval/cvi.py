import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist
from itertools import product
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import calinski_harabasz_score as chs
from sklearn.metrics import silhouette_score as ss
from sklearn.metrics import davies_bouldin_score as dbs

from pyclustkit.eval.index_specifics.s import return_s
from pyclustkit.eval.index_specifics.representatives import *
from pyclustkit.eval.index_specifics.FFT import fft
from pyclustkit.eval.index_specifics.sd import *


def dunn(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute Deltas
    Deltas = {i: pairwise_distances(x[y == i], x[y == i]).max() for i in clusters}
    # Compute deltas
    deltas = {(i, j): pairwise_distances(x[y == i], x[y == j]).min() for i, j in combinations(clusters, r=2)}
    return min(deltas.values()) / max(Deltas.values())


def gdi21(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute Deltas
    Deltas = {i: pairwise_distances(x[y == i], x[y == i]).max() for i in clusters}
    # Find centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    # Compute deltas
    deltas = {(i, j): pairwise_distances(x[y == i], x[y == j]).max() for i, j in combinations(clusters, r=2)}
    return min(deltas.values()) / max(Deltas.values())


def gdi31(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute Deltas
    Deltas = {i: pairwise_distances(x[y == i], x[y == i]).max() for i in clusters}
    # Find centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    # Compute deltas
    deltas = {(i, j): pairwise_distances(x[y == i], x[y == j]).sum() / (y[y == i].shape[0] * y[y == j].shape[0]) for
              i, j in combinations(clusters, r=2)}
    return min(deltas.values()) / max(Deltas.values())


def gdi41(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    # Compute Deltas
    Deltas = {i: pairwise_distances(x[y == i], x[y == i]).max() for i in clusters}
    # Find centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    # Compute deltas
    deltas = {(i, j): np.linalg.norm(centers[i] - centers[j]) for i, j in combinations(clusters, r=2)}
    return min(deltas.values()) / max(Deltas.values())


def gdi51(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    # Compute Deltas
    Deltas = {i: pairwise_distances(x[y == i], x[y == i]).max() for i in clusters}
    # Find centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    # Compute deltas
    deltas = {(i, j): (np.linalg.norm(x[y == i] - centers[i], axis=1).sum() + np.linalg.norm(x[y == j] - centers[j],
                                                                                             axis=1).sum()) / (
                              y[y == i].shape[0] + y[y == j].shape[0]) for i, j in combinations(clusters, r=2)}
    return min(deltas.values()) / max(Deltas.values())


def gdi61(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    # Compute Deltas
    Deltas = {i: pairwise_distances(x[y == i], x[y == i]).max() for i in clusters}
    # Compute deltas
    deltas = {(i, j): max(directed_hausdorff(x[y == i], x[y == j])[0], directed_hausdorff(x[y == j], x[y == i])[0]) for
              i, j in combinations(clusters, r=2)}
    return min(deltas.values()) / max(Deltas.values())


def gdi12(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute Deltas
    Deltas = {i: pdist(x[y == i]).sum() / (y[y == i].shape[0] * (y[y == i].shape[0] - 1)) for i in clusters}
    # Find centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    # Compute deltas
    deltas = {(i, j): pairwise_distances(x[y == i], x[y == j]).min() for i, j in combinations(clusters, r=2)}

    return min(deltas.values()) / max(Deltas.values())


def gdi22(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute Deltas
    Deltas = {i: pdist(x[y == i]).sum() / (y[y == i].shape[0] * (y[y == i].shape[0] - 1)) for i in clusters}
    # Find centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    # Compute deltas
    deltas = {(i, j): pairwise_distances(x[y == i], x[y == j]).max() for i, j in combinations(clusters, r=2)}
    return min(deltas.values()) / max(Deltas.values())


def gdi32(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute Deltas
    Deltas = {i: pdist(x[y == i]).sum() / (y[y == i].shape[0] * (y[y == i].shape[0] - 1)) for i in clusters}
    # Find centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    # Compute deltas
    deltas = {(i, j): pairwise_distances(x[y == i], x[y == j]).sum() / (y[y == i].shape[0] * y[y == j].shape[0]) for
              i, j in combinations(clusters, r=2)}
    return min(deltas.values()) / max(Deltas.values())


def gdi42(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    # Compute Deltas
    Deltas = {i: pdist(x[y == i]).sum() / (y[y == i].shape[0] * (y[y == i].shape[0] - 1)) for i in clusters}
    # Find centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    # Compute deltas
    deltas = {(i, j): np.linalg.norm(centers[i] - centers[j]) for i, j in combinations(clusters, r=2)}
    return min(deltas.values()) / max(Deltas.values())


def gdi52(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    # Compute Deltas
    Deltas = {i: pdist(x[y == i]).sum() / (y[y == i].shape[0] * (y[y == i].shape[0] - 1)) for i in clusters}
    # Find centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    # Compute deltas
    deltas = {(i, j): (np.linalg.norm(x[y == i] - centers[i], axis=1).sum() + np.linalg.norm(x[y == j] - centers[j],
                                                                                             axis=1).sum()) / (
                              y[y == i].shape[0] + y[y == j].shape[0]) for i, j in combinations(clusters, r=2)}
    return min(deltas.values()) / max(Deltas.values())


def gdi62(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    # Compute Deltas
    Deltas = {i: pdist(x[y == i]).sum() / (y[y == i].shape[0] * (y[y == i].shape[0] - 1)) for i in clusters}
    # Compute deltas
    deltas = {(i, j): max(directed_hausdorff(x[y == i], x[y == j])[0], directed_hausdorff(x[y == j], x[y == i])[0]) for
              i, j in combinations(clusters, r=2)}
    return min(deltas.values()) / max(Deltas.values())


def gdi13(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Find centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    # Compute Deltas
    Deltas = {i: np.linalg.norm(x[y == i] - centers[i], axis=1).sum() for i in clusters}

    Deltas = {i: np.linalg.norm(x[y == i] - centers[i], axis=1).sum() / (y[y == i].shape[0] / 2) for i in clusters}
    # Compute deltas
    deltas = {(i, j): pairwise_distances(x[y == i], x[y == j]).min() for i, j in combinations(clusters, r=2)}

    return min(deltas.values()) / max(Deltas.values())


def gdi23(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Find centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    # Compute Deltas
    Deltas = {i: np.linalg.norm(x[y == i] - centers[i], axis=1).sum() / (y[y == i].shape[0] / 2) for i in clusters}
    # Compute deltas
    deltas = {(i, j): pairwise_distances(x[y == i], x[y == j]).max() for i, j in combinations(clusters, r=2)}
    return min(deltas.values()) / max(Deltas.values())


def gdi33(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Find centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    # Compute Deltas
    Deltas = {i: np.linalg.norm(x[y == i] - centers[i], axis=1).sum() / (y[y == i].shape[0] / 2) for i in clusters}
    # Compute deltas
    deltas = {(i, j): pairwise_distances(x[y == i], x[y == j]).sum() / (y[y == i].shape[0] * y[y == j].shape[0]) for
              i, j in combinations(clusters, r=2)}
    return min(deltas.values()) / max(Deltas.values())


def gdi43(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    # Compute Deltas
    Deltas = {i: np.linalg.norm(x[y == i] - centers[i], axis=1).sum() / (y[y == i].shape[0] / 2) for i in clusters}
    # Compute deltas
    deltas = {(i, j): np.linalg.norm(centers[i] - centers[j]) for i, j in combinations(clusters, r=2)}
    return min(deltas.values()) / max(Deltas.values())


def gdi53(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    # Compute Deltas
    Deltas = {i: np.linalg.norm(x[y == i] - centers[i], axis=1).sum() / (y[y == i].shape[0] / 2) for i in clusters}
    # Compute deltas
    deltas = {(i, j): (np.linalg.norm(x[y == i] - centers[i], axis=1).sum() + np.linalg.norm(x[y == j] - centers[j],
                                                                                             axis=1).sum()) / (
                              y[y == i].shape[0] + y[y == j].shape[0]) for i, j in combinations(clusters, r=2)}
    return min(deltas.values()) / max(Deltas.values())


def gdi63(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    # Compute Deltas
    Deltas = {i: np.linalg.norm(x[y == i] - centers[i], axis=1).sum() / (y[y == i].shape[0] / 2) for i in clusters}
    # Compute deltas
    deltas = {(i, j): max(directed_hausdorff(x[y == i], x[y == j])[0], directed_hausdorff(x[y == j], x[y == i])[0]) for
              i, j in combinations(clusters, r=2)}
    return min(deltas.values()) / max(Deltas.values())


def trace_wib(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    X_cen = x.mean(axis=0)
    B_mat = []
    for i in range(x.shape[0]):
        B_mat.append(centers[y[i]] - X_cen)
    B_mat = np.array(B_mat)
    BG_mat = np.matmul(B_mat.T, B_mat)
    WG = np.zeros((x.shape[1], x.shape[1]))
    X_cl = {i: x[y == i] - centers[i] for i in clusters}
    WG_k_mat = {i: np.matmul(X_cl[i].T, X_cl[i]) for i in clusters}
    WG_mat = np.zeros((x.shape[1], x.shape[1]))
    for i in WG_k_mat.values():
        WG_mat += i
    return np.matmul(np.linalg.inv(WG_mat), BG_mat).trace()


def trace_w(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute Z matrix
    Z = np.zeros((x.shape[0], len(clusters)))
    for i in range(x.shape[0]):
        Z[i, y[i]] = 1
    # Compute clusters' centers
    X_bar = np.matmul(np.matmul(np.linalg.inv(np.matmul(Z.T, Z)), Z.T), x)
    # Compute between-cluster SSCP matrix
    B = np.matmul(np.matmul(np.matmul(X_bar.T, Z.T), Z), X_bar)
    # Compute total-sample sum-of-squares and crossproducts SSCP matrix
    T = np.matmul(x.T, x)
    # Compute within-cluster SSCP matrix
    W = T - B
    return W.trace()


def ball_hall(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    sum_M_G = {i: np.array([np.linalg.norm(x[y == i] - centers[i]) ** 2]).sum() for i in clusters}

    return np.array([sum_M_G[i] / y[y == i].shape[0] for i in clusters]).mean()


def c_index(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    n_k = {i: y[y == i].shape[0] for i in clusters}
    N_w = int(np.array([n_k[i] * (n_k[i] - 1) / 2 for i in clusters]).sum())

    S_w = {i: sum(pdist(x[y == i])) for i in clusters}

    S_w = np.array([S_w[i] for i in clusters]).sum()

    dist = pdist(x)
    dist.sort()
    dist = np.array(dist)
    S_min = dist[:N_w].sum()
    S_max = dist[dist.shape[0] - N_w:].sum()
    return (S_w - S_min) / (S_max - S_min)


def mcclain_rao(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute numerator
    num = 0
    step = 0
    for cluster in clusters:
        temp = x[y == cluster]
        for i in range(temp.shape[0] - 1):
            for j in range(i + 1, temp.shape[0]):
                num += np.linalg.norm(temp[i] - temp[j])
                step += 1
    num /= step

    # Compute denominator
    den = 0
    step = 0
    for cluster_i in range(len(clusters) - 1):
        temp_1 = x[y == clusters[cluster_i]]
        for i in temp_1:
            for cluster_l in range(cluster_i + 1, len(clusters)):
                temp_2 = x[y == cluster_l]
                for j in temp_2:
                    den += np.linalg.norm(i - j)
                    step += 1
    den /= step
    return num / den


def ratkowsky_lance(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    X_cen = x.mean(axis=0)
    BGSS = np.zeros((1, x.shape[1]))
    for j in range(BGSS.shape[1]):
        for cluster in clusters:
            BGSS[0, j] += y[y == cluster].shape[0] * (centers[cluster][j] - X_cen[j]) ** 2

    # TSS = np.array([((x[:,j] - X_cen[j])**2).sum() for j in range(x.shape[1])])
    TSS = np.zeros((1, x.shape[1]))
    for j in range(TSS.shape[1]):
        for i in range(x.shape[0]):
            TSS[0, j] += (x[i, j] - X_cen[j]) ** 2

    R_bar = (BGSS / TSS).mean()
    return math.sqrt(R_bar / len(clusters))


def pbm(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    D = {(i, j): np.linalg.norm(centers[i] - centers[j]) for i, j in combinations(clusters, r=2)}
    D_B = np.array(list(D.values())).max()
    E_W = 0
    for i in clusters:
        for vector in x[y == i]:
            E_W += np.linalg.norm(vector - centers[i])
    X_cen = x.mean(axis=0)
    E_T = np.linalg.norm(x - X_cen, axis=1).sum()
    return ((1 / len(clusters)) * (E_T / E_W) * D_B) ** 2


def friedman_rudin_1(x, y):
    # TODO: same as W-1B
    # Find distinct clusters
    clusters = list(set(y))
    # Compute Z matrix
    Z = np.zeros((x.shape[0], len(clusters)))
    for i in range(x.shape[0]):
        Z[i, y[i]] = 1
    # Compute clusters' centers
    X_bar = np.matmul(np.matmul(np.linalg.inv(np.matmul(Z.T, Z)), Z.T), x)
    # Compute between-cluster SSCP matrix
    B = np.matmul(np.matmul(np.matmul(X_bar.T, Z.T), Z), X_bar)
    # Compute total-sample sum-of-squares and crossproducts SSCP matrix
    T = np.matmul(x.T, x)
    # Compute within-cluster SSCP matrix
    W = T - B
    return (np.dot(np.linalg.inv(W), B)).trace()


def sd_scat(x, y):
    clusters = list(set(y))
    per_cluster_var = {i: np.var(x[y==i], axis=0) for i in clusters}

    total_var = np.var(x, axis=0)

    return ((sum({i: np.linalg.norm(j)  for i,j in per_cluster_var.items()}.values()) / len(clusters))/
            np.linalg.norm(total_var))




def friedman_rudin_2(x, y):
    X_cen = x.mean(axis=0)
    x = x - X_cen
    # Find distinct clusters
    clusters = list(set(y))
    Z = np.zeros((x.shape[0], len(clusters)))
    for i in range(x.shape[0]):
        Z[i, y[i]] = 1
    # Compute clusters' centers
    X_bar = np.matmul(np.matmul(np.linalg.inv(np.matmul(Z.T, Z)), Z.T), x)
    # Compute between-cluster SSCP matrix
    B = np.matmul(np.matmul(np.matmul(X_bar.T, Z.T), Z), X_bar)
    # Compute total-sample sum-of-squares and crossproducts SSCP matrix
    T = np.matmul(x.T, x)
    # Compute within-cluster SSCP matrix
    W = T - B
    return np.linalg.norm(T) / np.linalg.norm(W)


def log_ss_ratio(x, y):
    clusters = list(set(y))

    # Cluster centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    # Within group dispersion
    WG_k = {i: np.array([np.linalg.norm(vector - centers[i]) ** 2 for vector in x[y == i]]).sum() for i in clusters}
    WGSS = np.array(list(WG_k.values())).sum()

    X_cen = x.mean(axis=0)
    BGSS = np.array([y[y == i].shape[0] * np.linalg.norm(centers[i] - X_cen) ** 2 for i in clusters]).sum()
    return np.log(BGSS / WGSS)


def log_det_ratio(x, y):
    X_cen = x.mean(axis=0)
    x = x - X_cen
    # Find distinct clusters
    clusters = list(set(y))
    Z = np.zeros((x.shape[0], len(clusters)))
    for i in range(x.shape[0]):
        Z[i, y[i]] = 1
    # Compute clusters' centers
    X_bar = np.matmul(np.matmul(np.linalg.inv(np.matmul(Z.T, Z)), Z.T), x)
    # Compute between-cluster SSCP matrix
    B = np.matmul(np.matmul(np.matmul(X_bar.T, Z.T), Z), X_bar)
    # Compute total-sample sum-of-squares and crossproducts SSCP matrix
    T = np.matmul(x.T, x)
    # Compute within-cluster SSCP matrix
    W = T - B

    return x.shape[0] * np.log(np.linalg.det(T) / np.linalg.det(W))


def sd_dis(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    # Compute distances between centers
    centers_distances = {(i, j): np.linalg.norm(centers[i] - centers[j]) for i, j in list(product(clusters, clusters))}

    # Compute Dmax
    Dmax = np.array(list(centers_distances.values())).max()
    # Compute Dmin
    Dmin = np.delete(np.array(list(centers_distances.values())),
                     np.where(np.array(list(centers_distances.values())) == 0)).min()
    # Compute total separation between clusters
    Dis = 0
    for i in clusters:
        s = 0
        for j in clusters:
            if i != j:
                s += centers_distances[(i, j)]
        Dis += s ** (-1)

    Dis = (Dmax / Dmin) * Dis
    return Dis


def det_ratio(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    WG = np.zeros((x.shape[1], x.shape[1]))
    X_cen = x.mean(axis=0)
    X_cl = {i: x[y == i] - centers[i] for i in clusters}
    WG_k_mat = {i: np.matmul(X_cl[i].T, X_cl[i]) for i in clusters}
    WG_mat = np.zeros((x.shape[1], x.shape[1]))
    for i in WG_k_mat.values():
        WG_mat += i
    X_ds = x - X_cen
    T_mat = np.matmul(X_ds.T, X_ds)
    return np.linalg.det(T_mat) / np.linalg.det(WG_mat)


def ray_turi(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    WG_k = {i: np.array([np.linalg.norm(vector - centers[i]) ** 2 for vector in x[y == i]]).sum() for i in clusters}
    WGSS = np.array(list(WG_k.values())).sum()
    Deltas = {(i, j): np.linalg.norm(centers[i] - centers[j]) ** 2 for i, j in combinations(clusters, r=2)}
    return (WGSS / x.shape[0]) / np.array(list(Deltas.values())).min()


def xie_beni(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    WG_k = {i: np.array([np.linalg.norm(vector - centers[i]) ** 2 for vector in x[y == i]]).sum() for i in clusters}
    WGSS = np.array(list(WG_k.values())).sum()
    deltas = {(i, j): pairwise_distances(x[y == i], x[y == j]).min() ** 2 for i, j in combinations(clusters, r=2)}

    return WGSS / (np.array(list(deltas.values())).min() * x.shape[0])


def wemmert_gancarski(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    R = [np.linalg.norm(x[i] - centers[y[i]]) / np.array(
        [np.linalg.norm(x[i] - centers[j]) for j in clusters if j != y[i]]).min() for i in range(x.shape[0])]
    R = np.array(R)
    J_k = {i: max(0, 1 - (R[y == i].sum() / y[y == i].shape[0])) for i in clusters}

    return np.array([y[y == i].shape[0] * J_k[i] for i in clusters]).sum() / x.shape[0]


def banfeld_raftery(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    s = 0
    for cluster in clusters:
        s += y[y == cluster].shape[0] * np.log(
            (np.linalg.norm(x[y == cluster] - centers[cluster], axis=1) ** 2).sum() / y[y == cluster].shape[0])
    return s


def ksq_detw(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    Z = np.zeros((x.shape[0], len(clusters)))
    for i in range(x.shape[0]):
        Z[i, y[i]] = 1
    # Compute clusters' centers
    X_bar = np.matmul(np.matmul(np.linalg.inv(np.matmul(Z.T, Z)), Z.T), x)
    # Compute between-cluster SSCP matrix
    B = np.matmul(np.matmul(np.matmul(X_bar.T, Z.T), Z), X_bar)
    # Compute total-sample sum-of-squares and crossproducts SSCP matrix
    T = np.matmul(x.T, x)
    # Compute within-cluster SSCP matrix
    W = T - B
    return len(clusters) ** 2 * np.linalg.det(W)


def scott_symons(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute centers
    centers = {i: x[y == i].mean(axis=0) for i in clusters}
    # WG_k = {i:np.array([np.linalg.norm(vector - centers[i])**2 for vector in x[y==i]]).sum() for i in clusters}
    # WG_k = {i:np.zeros((x[y==i].shape[0], x[y==i].shape[1])) for i in clusters}
    X = {i: x[y == i] - centers[i] for i in clusters}
    WG_k = {i: np.matmul(X[i].T, X[i]) for i in clusters}
    return np.array([y[y == i].shape[0] * np.log(np.linalg.det(WG_k[i] / y[y == i].shape[0])) for i in clusters]).sum()


def point_biserial(x, y):
    # Find distinct clusters
    clusters = list(set(y))
    # Compute S_W / N_w
    num = 0
    stepw = 0
    for cluster in clusters:
        temp = x[y == cluster]
        for i in range(temp.shape[0] - 1):
            for j in range(i + 1, temp.shape[0]):
                num += np.linalg.norm(temp[i] - temp[j])
                stepw += 1

    num /= stepw
    # Compute S_B / N_B
    den = 0
    stepb = 0
    for cluster_i in range(len(clusters) - 1):
        temp_1 = x[y == clusters[cluster_i]]
        for i in temp_1:
            for cluster_l in range(cluster_i + 1, len(clusters)):
                temp_2 = x[y == cluster_l]
                for j in temp_2:
                    den += np.linalg.norm(i - j)
                    stepb += 1
    den /= stepb
    N_T = x.shape[0] * (x.shape[0] - 1) / 2

    return (num - den) * math.sqrt(stepw * stepb) / N_T


def calinski_harabasz(x, y):
    return chs(x, y)


def silhouette(x, y):
    return ss(x, y)


def davies_bouldin(x, y):
    return dbs(x, y)


def s_dbw(X, y):
    clusters = np.unique(y)
    scat_ = sd_scat(X, y)

    # calculate sigma - the radius of the hypersphere to determine neighborhood population.
    cstd = {}
    for clust in clusters:
        cstd[clust] = np.var(X[y == clust], axis=0)

    sigma = sum([np.linalg.norm(x) for x in cstd.values()])
    sigma = math.sqrt(sigma) / len(clusters)

    dens_bw_ = dens_bw(X, y, sigma)
    return scat_ + dens_bw_


def cdbw(X, y):
    """

    Args:
        X:
        y:

    Returns:

    """

    representatives = fft(X, y, 5)
    coh_, compactness_ = cohesion(X, y, reps=representatives, cluster_centers(x, y))

    cr = closest_representatives(X, representatives)
    rcr = respective_closest_representatives(representatives, cr)
    dens = density(X, y, rcr)
    inter_dens = inter_density(dens)

    sep = cluster_separation(X, rcr, inter_dens)

    sc = sep * compactness_

    return sc * coh_


def gamma(X, y, precomputed_distances=False):
    if not precomputed_distances:

        s_plus, s_minus, nb, nw = return_s(X, y, precomputed_distances=False)
    else:
        assert type(precomputed_distances) is np.array, ("if precomputed_distances=False, an np.array of pairwise "
                                                         "distances should be provided.")
        s_plus, s_minus, nb, nw = return_s(X, y)

    return (s_plus - s_minus) / (s_plus + s_minus)


def tau(X, y, precomputed_distances=False):
    if not precomputed_distances:

        s_plus, s_minus, nb, nw = return_s(X, y, precomputed_distances=False)
    else:
        assert type(precomputed_distances) is np.array, ("if precomputed_distances=False, an np.array of pairwise "
                                                         "distances should be provided.")
        s_plus, s_minus, nb, nw = return_s(X, y)

    return (s_plus - s_minus) / (nb * nw * ((X.shape[0] * (X.shape[0] - 1)) / 2)) * 1 / 2
