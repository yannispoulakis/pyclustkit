import numpy as np
from copy import deepcopy
from pyclustkit.eval.core._shared_processes import process_adg
from pyclustkit.eval.core._adg_operations import execute_graph, get_subgraph, topological_sort
from pyclustkit.eval.core._shared_processes import common_operations
from pyclustkit.eval.core._utils import find_midpoint
from itertools import combinations
from scipy.spatial.distance import cdist
from pyclustkit.eval.index_specifics.representatives import *
from pyclustkit.eval.index_specifics.FFT import fft
import traceback




# TODO: Write Hausdorff in simpler terms
# TODO: Write functions for deltas denominators
# TODO: No citation for wemmert Gancarski, has it been published? (research)
# TODO: Check for more indices other than R - ClusterCrit


class CVIToolbox:
    cvi_opt_type = {"max": ['calinski_harabasz', 'cdbw', 'dunn', 'gamma', 'gdi12', 'gdi13', 'gdi21', 'gdi22',
                            'gdi23', 'gdi31', 'gdi32', 'gdi33', 'gdi41', 'gdi42', 'gdi43', 'gdi51', 'gdi52',
                            'gdi53', 'gdi61', 'gdi62', 'gdi63', 'pbm', 'point_biserial', 'ratkowsky_lance',
                            'silhouette', 'tau', 'wemmert_gancarski'],
                    "min": ['banfeld_raftery', 'c_index', 'davies_bouldin', 'g_plus', 'mcclain_rao',
                            'ray_turi', 'scott_symons', 's_dbw', 'xie_beni'],
                    "max_diff": ['ball_hall', 'ksq_detw', 'trace_w', 'trace_wib'],
                    "min_diff": ['det_ratio', 'log_det_ratio', 'log_ss_ratio']}

    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.noclusters = len(np.unique(self.y))

        assert self.X.shape[0] == len(self.y), "Dataset instances should match number of labels."

        self.cvi_higher_best = {'dunn': True}
        self.processes = {}
        self.processes = deepcopy(common_operations)
        self.processes['x']['value'] = self.X
        self.processes['y']['value'] = self.y
        self.cvi_results = {}
        self.cvi_methods_list = {'ball_hall': self.ball_hall, 'banfeld_raftery': self.banfeld_raftery,
                                 'c_index': self.c_index, 'calinski_harabasz': self.calinski_harabasz,
                                 'cdbw': self.cdbw,
                                 'davies_bouldin': self.davies_bouldin, 'det_ratio': self.det_ratio, 'dunn': self.dunn,
                                 'g_plus': self.g_plus, 'gamma': self.gamma,
                                 'gdi11': self.gdi11, 'gdi12': self.gdi12, 'gdi13': self.gdi13,
                                 'gdi21': self.gdi21, 'gdi22': self.gdi22, 'gdi23': self.gdi23,
                                 'gdi31': self.gdi31, 'gdi32': self.gdi32, 'gdi33': self.gdi33,
                                 'gdi41': self.gdi41, 'gdi42': self.gdi42, 'gdi43': self.gdi43,
                                 'gdi51': self.gdi51, 'gdi52': self.gdi52, 'gdi53': self.gdi53,
                                 'gdi61': self.gdi61, 'gdi62': self.gdi62, 'gdi63': self.gdi63,
                                 'ksq_detw': self.ksq_detw,
                                 'log_det_ratio': self.log_det_ratio, 'log_ss_ratio': self.log_ss_ratio,
                                 'mcclain_rao': self.mcclain_rao, 'pbm': self.pbm,
                                 'point_biserial': self.point_biserial,
                                 'ray_turi': self.ray_turi, 'ratkowsky_lance': self.ratkowsky_lance,
                                 'scott_symons': self.scott_symons, 'sd_scat': self.sd_scat, 'sd_dis': self.sd_dis,
                                 's_dbw': self.sdbw, 'silhouette': self.silhouette, 'trace_w': self.trace_w,
                                 'wemmert_gancarski': self.wemmert_gancarski, 'xie_beni': self.xie_beni,
                                 'trace_wib': self.trace_wib,
                                 'tau': self.tau}



    def calculate_icvi(self, cvi='all', exclude=None):
        """
        The base method for searching the subset of CVI to be calculated and applying the necessary methods.

        Args:
            exclude (None or list): The CVI to exclude from calculation
            cvi (string or list): Can be either <all> or a list of provided indexes ['dunn','c_index', …]

        Raises:
            AssertionError: When the list of cvi provided includes CVI with typos or indexes not yet implemented
            TypeError: When both the cvi and exclude parameters are lists

        Returns:
            None: Does not return anything, instead use CVIToolbox.cvi_results

        Examples:
            (1) General usage
            >> cvit = CVIToolbox(X, y)
            >> cvit.calculate_icvi()
        """
        # Assertions and errors.
        if type(cvi) is list:
            assert all(x in self.cvi_methods_list.keys() for x in cvi), ("provided indexes may be misspelled or not yet"
                                                                         " implemented")
        if type(exclude) is list:
            assert all(x in self.cvi_methods_list.keys() for x in exclude), ("provided indexes to exclude may be "
                                                                             "misspelled or not yet implemented")
        if type(exclude) is list and type(cvi) is list:
            raise TypeError("Either cvi or exclude must be a list, not both.")

        # Computation
        if cvi == 'all':
            if exclude is not None:
                cvi_methods_dict = {k: v for k, v in self.cvi_methods_list.items() if k not in exclude}
            else:
                cvi_methods_dict = self.cvi_methods_list
            for index in cvi_methods_dict.keys():
                try:
                    self.cvi_results[index] = self.cvi_methods_list[index]()
                except Exception as e:
                    print(f'{index}: {e}')
                    traceback.print_exc()
        else:
            for index in cvi:
                try:
                    self.cvi_results[index] = self.cvi_methods_list[index]()
                except Exception as e:
                    print(f'{index}: {e}')
                    traceback.print_exc()

    def execute_subprocess(self, subprocess):
        sg = get_subgraph(process_adg, subprocess)
        eo = topological_sort(sg)
        process_results = execute_graph(eo, self.processes)
        for key in process_results.keys():
            self.processes[key]['value'] = process_results[key]
        return process_results[subprocess]

    def silhouette(self):
        labels = np.unique(self.y)
        idxs = {lab: np.flatnonzero(self.y == lab) for lab in labels}
        intra_dist = self.execute_subprocess("intra_cluster_distances")

        # a(i)
        intra_dist = {x: np.sum(y, axis=1) / (y.shape[1] - 1) for x, y in intra_dist.items()}

        inter_dist = self.execute_subprocess("inter_cluster_distances")
        # b(i)
        per_point_min_dist_of_nearest = {}
        for label in set(self.y):
            relevant_keys = [key for key in inter_dist.keys() if label in key]

            # First, we find the mean distance of one point to all other points for each cluster
            per_point_min_inter_dist = []
            for comb in relevant_keys:
                dist_df = inter_dist[comb]
                if comb[0] == label:
                    mean_arr = np.mean(dist_df, axis=1).reshape(-1, 1)
                else:
                    mean_arr = np.mean(dist_df, axis=0).reshape(-1, 1)
                per_point_min_inter_dist.append(mean_arr)
            per_point_min_dist_of_nearest[label] = np.hstack(per_point_min_inter_dist)

        sum = 0
        counter = 0
        for key in intra_dist:
            for row in range(0, intra_dist[key].shape[0]):
                counter += 1
                a_i = intra_dist[key][row]
                b_i = np.min(per_point_min_dist_of_nearest[key][row])
                s_i = (b_i - a_i) / (np.max([b_i, a_i]))
                sum += s_i
        return sum / counter

    def calinski_harabasz(self):
        wgd = self.execute_subprocess("within_group_dispersion")
        bgd = self.execute_subprocess("between_group_dispersion")
        self.cvi_results["calinski_harabasz"] = (bgd / wgd) * (
                    (self.X.shape[0] - len(set(self.y))) / (len(set(self.y)) - 1))
        return self.cvi_results["calinski_harabasz"]

    def davies_bouldin(self):
        pcenters_dist = self.execute_subprocess("pairwise_cluster_centers_distances")
        sdfcc = self.execute_subprocess("sum_distances_from_cluster_centers")
        for key in sdfcc:
            sdfcc[key] = sdfcc[key] / len(self.y[self.y == key])

        max_rij_sum = 0
        for key in sdfcc:
            relevant_combs = [tup for tup in list(combinations(sdfcc.keys(), r=2)) if key in tup]
            ri_j = []
            for comb in relevant_combs:
                clust_dist = pcenters_dist[comb]
                si_plus_sj = sdfcc[comb[0]] + sdfcc[comb[1]]
                ri_j.append(si_plus_sj / clust_dist)
            max_rij_sum += max(ri_j)

        db = max_rij_sum / len(sdfcc.keys())

        self.cvi_results["davies_bouldin"] = db
        return db

    def ball_hall(self):
        """
         - G. H. Ball and D. J. Hall. Isodata: A novel method of data analysis and pattern classification. Menlo Park:
         Stanford Research Institute. (NTIS No. AD 699616), 1965.

        Returns:

        """
        d_from_ccenters = self.execute_subprocess('distances_from_cluster_centers')
        d_from_ccenters = {i: (j ** 2).sum() for i, j in d_from_ccenters.items()}
        return np.array([d_from_ccenters[i] / self.y[self.y == i].shape[0] for i in np.unique(self.y)]).mean()

    def banfeld_raftery(self):
        wgk = self.execute_subprocess('within_group_scatter_matrices')
        br = {i: len(self.y[self.y == i]) * np.log(np.trace(j) / len(self.y[self.y == i])) for i, j in wgk.items()}
        return sum(br.values())

    def c_index(self):
        """

        -Hubert, Lawrence & Levin, Joel. (1976). A general statistical framework for assessing categorical clustering in
        free recall. Psychological Bulletin. 83. 1072-1080. 10.1037/0033-2909.83.6.1072.

        :return: (float) Value of the C-index
        """
        s_w = self.execute_subprocess('sum_intra_cluster_distances')
        s_w = np.sum(list(s_w.values()))

        n_w = self.execute_subprocess('intra_cluster_distances')
        n_w = sum([(j.shape[0] * (j.shape[0] - 1)) / 2 for i, j in n_w.items()])
        n_w = int(n_w)

        p_distances = self.execute_subprocess('pairwise_distances')
        m = p_distances.shape[0]
        r, c = np.triu_indices(m, 1)
        p_distances = np.sort(p_distances[r, c])
        s_min = sum(p_distances[:n_w])
        s_max = sum(p_distances[-n_w:])

        return (s_w - s_min) / (s_max - s_min)

    def cdbw(self):
        """
        -Halkidi et al. On clustering validation techniques. J. Intell.nf. Syst., 17(2-3):107–145, 2001.
        = cohesion * SC
        where:
            -cohesion=

        Returns:

        """
        cc = self.execute_subprocess("cluster_centers")

        reps = fft(self.X, self.y, 5)
        coh, compactness = cohesion(self.X, self.y, reps, cc)

        cr = closest_representatives(self.X, reps)
        rcr = respective_closest_representatives(reps, cr)
        dens = density(self.X, self.y, rcr)
        inter_dens = inter_density(dens)
        sep = cluster_separation(self.X, rcr, inter_dens)
        sc = sep * compactness
        return sc * coh

    def det_ratio(self):
        """
        - A. J. Scott and M. J. Symons. Clustering methods based on likelihood ratio criteria. Biometrics,
        27:387–397, 1971.
        = det(Total scatter matrix)/det(within group scatter matrix)
        Returns:

        """
        WG = self.execute_subprocess('within_group_scatter_matrix')

        T = self.execute_subprocess('total_scatter_matrix')

        return np.linalg.det(T) / np.linalg.det(WG)

    def dunn(self):
        """
        - J. Dunn. Well separated clusters and optimal fuzzy partitions. Journal of Cybernetics, 4:95–104, 1974.
        = dmin/ dmax
        closest and furthest distances between clusters
        Returns:

        """
        deltas = self.execute_subprocess('min_inter_cluster_distances')
        Deltas = self.execute_subprocess('max_intra_cluster_distances')
        return min(deltas.values()) / max(Deltas.values())

    def g_plus(self):
        s_plus, s_minus, nb, nw = self.execute_subprocess('s_values')
        return (2 * s_minus) / ((self.X.shape[0] * (self.X.shape[0] - 1)) / 2)

    def gamma(self):
        s_plus, s_minus, nb, nw = self.execute_subprocess('s_values')
        return (s_plus - s_minus) / (s_plus + s_minus)

    def gdi11(self):
        Deltas = self.execute_subprocess('max_intra_cluster_distances')
        deltas = self.execute_subprocess('min_inter_cluster_distances')
        return min(deltas.values()) / max(Deltas.values())

    def gdi21(self):
        Deltas = self.execute_subprocess('max_intra_cluster_distances')
        deltas = self.execute_subprocess('max_inter_cluster_distances')
        return min(deltas.values()) / max(Deltas.values())

    def gdi31(self):
        Deltas = self.execute_subprocess('max_intra_cluster_distances')
        deltas = self.execute_subprocess('sum_inter_cluster_distances')

        deltas = {i: j / (len(self.y[self.y == i[0]]) * len(self.y[self.y == i[1]])) for i, j in deltas.items()}
        return min(deltas.values()) / max(Deltas.values())

    def gdi41(self):
        Deltas = self.execute_subprocess('max_intra_cluster_distances')
        deltas = self.execute_subprocess('pairwise_cluster_centers_distances')

        return min(deltas.values()) / max(Deltas.values())

    def gdi51(self):
        Deltas = self.execute_subprocess('max_intra_cluster_distances')
        deltas = self.execute_subprocess('pairwise_sum_distances_from_cluster_centers')
        deltas = {i: j / (len(self.y[self.y == i[0]]) + len(self.y[self.y == i[1]])) for i, j in deltas.items()}
        return min(deltas.values()) / max(Deltas.values())

    def gdi61(self):
        Deltas = self.execute_subprocess('max_intra_cluster_distances')
        deltas = self.execute_subprocess('pairwise_hausdorff')

        return min(deltas.values()) / max(Deltas.values())

    def gdi12(self):
        Deltas = self.execute_subprocess('sum_intra_cluster_distances')
        Deltas = {i: j / (len(self.y[self.y == i]) * (len(self.y[self.y == i]) - 1)) for i, j in Deltas.items()}
        deltas = self.execute_subprocess('min_inter_cluster_distances')
        return min(deltas.values()) / max(Deltas.values())

    def gdi22(self):
        Deltas = self.execute_subprocess('sum_intra_cluster_distances')
        Deltas = {i: j / (len(self.y[self.y == i]) * (len(self.y[self.y == i]) - 1)) for i, j in Deltas.items()}
        deltas = self.execute_subprocess('max_inter_cluster_distances')

        return min(deltas.values()) / max(Deltas.values())

    def gdi32(self):
        Deltas = self.execute_subprocess('sum_intra_cluster_distances')
        Deltas = {i: j / (len(self.y[self.y == i]) * (len(self.y[self.y == i]) - 1)) for i, j in Deltas.items()}
        deltas = self.execute_subprocess('sum_inter_cluster_distances')
        deltas = {i: j / (len(self.y[self.y == i[0]]) * len(self.y[self.y == i[1]])) for i, j in deltas.items()}

        return min(deltas.values()) / max(Deltas.values())

    def gdi42(self):
        Deltas = self.execute_subprocess('sum_intra_cluster_distances')
        Deltas = {i: j / (len(self.y[self.y == i]) * (len(self.y[self.y == i]) - 1)) for i, j in Deltas.items()}
        deltas = self.execute_subprocess('pairwise_cluster_centers_distances')
        return min(deltas.values()) / max(Deltas.values())

    def gdi52(self):
        Deltas = self.execute_subprocess('sum_intra_cluster_distances')
        Deltas = {i: j / (len(self.y[self.y == i]) * (len(self.y[self.y == i]) - 1)) for i, j in Deltas.items()}
        deltas = self.execute_subprocess('pairwise_sum_distances_from_cluster_centers')
        deltas = {i: j / (len(self.y[self.y == i[0]]) + len(self.y[self.y == i[1]])) for i, j in deltas.items()}
        return min(deltas.values()) / max(Deltas.values())

    def gdi62(self):
        Deltas = self.execute_subprocess('sum_intra_cluster_distances')
        Deltas = {i: j / (self.y[self.y == i].shape[0] * (self.y[self.y == i].shape[0] - 1)) for i, j in Deltas.items()}
        deltas = self.execute_subprocess('pairwise_hausdorff')
        return min(deltas.values()) / max(Deltas.values())

    def gdi13(self):
        Deltas = self.execute_subprocess('sum_distances_from_cluster_centers')
        Deltas = {i: j / (len(self.y[self.y == i]) / 2) for i, j in Deltas.items()}
        deltas = self.execute_subprocess('min_inter_cluster_distances')
        return min(deltas.values()) / max(Deltas.values())

    def gdi23(self):
        Deltas = self.execute_subprocess('sum_distances_from_cluster_centers')
        Deltas = {i: j / (len(self.y[self.y == i]) / 2) for i, j in Deltas.items()}
        deltas = self.execute_subprocess('max_inter_cluster_distances')
        return min(deltas.values()) / max(Deltas.values())

    def gdi33(self):
        Deltas = self.execute_subprocess('sum_distances_from_cluster_centers')
        Deltas = {i: j / (len(self.y[self.y == i]) / 2) for i, j in Deltas.items()}
        deltas = self.execute_subprocess('sum_inter_cluster_distances')
        deltas = {i: j / (len(self.y[self.y == i[0]]) * len(self.y[self.y == i[1]])) for i, j in deltas.items()}
        return min(deltas.values()) / max(Deltas.values())

    def gdi43(self) -> float:
        """
        https://ieeexplore.ieee.org/document/8710320
        :return: Value of GDI43 (float)
        """
        Deltas = self.execute_subprocess('sum_distances_from_cluster_centers')
        Deltas = {i: j / (len(self.y[self.y == i]) / 2) for i, j in Deltas.items()}
        deltas = self.execute_subprocess('pairwise_cluster_centers_distances')
        return min(deltas.values()) / max(Deltas.values())

    def gdi53(self) -> float:
        """
        https://ieeexplore.ieee.org/document/8710320
        :return:
        """
        deltas = self.execute_subprocess('pairwise_sum_distances_from_cluster_centers')

        deltas = {i: j / (len(self.y[self.y == i[0]]) + len(self.y[self.y == i[1]])) for i, j in deltas.items()}

        Deltas = self.execute_subprocess('sum_distances_from_cluster_centers')
        Deltas = {i: 2 * j / len(self.y[self.y == i]) for i, j in Deltas.items()}

        return min(deltas.values()) / max(Deltas.values())

    def gdi63(self):
        Deltas = self.execute_subprocess('sum_distances_from_cluster_centers')
        Deltas = {i: j / (len(self.y[self.y == i]) / 2) for i, j in Deltas.items()}
        deltas = self.execute_subprocess('pairwise_hausdorff')
        return min(deltas.values()) / max(Deltas.values())

    def ksq_detw(self):
        """
        -  F. H. B. Marriot. Practical problems in a method of cluster analysis. Biometrics,
            27:456–460, 1975.

        = K^2 * det(WG)

        Returns:
            float
        """
        k2 = len(np.unique(self.y)) ** 2
        wg = self.execute_subprocess('within_group_scatter_matrix')
        detwg = np.linalg.det(wg)

        return k2 * detwg

    def log_det_ratio(self):
        """
        - Halkidi et al. On clustering validation techniques. J. Intell. Inf. Syst., 2001.
        = N*log(det(Total scatter matrix)/det(within group scatter matrix))
        Returns:

        """
        if 'det_ratio' in self.cvi_results.keys():
            return self.X.shape[0] * np.log(self.cvi_results['det_ratio'])

        WG = self.execute_subprocess('within_group_scatter_matrix')

        T = self.execute_subprocess('total_scatter_matrix')
        return self.X.shape[0] * np.log(np.linalg.det(T) / np.linalg.det(WG))

    def log_ss_ratio(self):
        """
        =log(between group dispersion/within group dispersion)
        J. A. Hartigan. Clustering algorithms. New York: Wiley, 1975.

        Returns:
            float

        """
        BGSS = self.execute_subprocess('between_group_dispersion')
        WGSS = self.execute_subprocess('within_group_dispersion')
        return np.log(BGSS / WGSS)

    def mcclain_rao(self):
        """
        - J. O. McClain and V. R. Rao. Clustisz: A program to test for the quality of
            clustering of a set of objects. Journal of Marketing Research, 12:456–460, 1975.
        Returns:

        """

        numerator = self.execute_subprocess('sum_intra_cluster_distances')

        numerator = {i: j / (len(self.y[self.y == i]) * (len(self.y[self.y == i]) - 1) / 2) for i, j in
                     numerator.items()}

        numerator = np.mean(list(numerator.values()))
        denominator = self.execute_subprocess('sum_inter_cluster_distances')

        denominator = {i: j / (len(self.y[self.y == i[0]]) * len(self.y[self.y == i[1]])) for i, j in
                       denominator.items()}

        denominator = np.mean(list(denominator.values()))
        return numerator / denominator

    def pbm(self):
        """
        - Bandyopadhyay S. Pakhira M. K. and Maulik U. Validity index for crisp and fuzzy clusters.
        DB : maximum of the cluster centers distances
        EW : the sum of the distances of the points of each cluster to their barycenter
        ET : the sum of the distances of all the points to the barycenter G of the entire data set
        K : The number of clusters
        :return:
        """

        DB = self.execute_subprocess('pairwise_cluster_centers_distances')
        DB = max(DB.values())

        EW = self.execute_subprocess('distances_from_cluster_centers')
        EW = {i: j.sum() for i, j in EW.items()}
        EW = sum(EW.values())

        ET = self.execute_subprocess('distances_from_data_center')
        ET = ET.sum()

        K = len(np.unique(self.y))

        pbm = ((1 / K) * (ET / EW) * DB) ** 2
        return pbm

    def point_biserial(self):
        """
        - G. W. Milligan. A monte carlo study of thirty internal criterion measures for
          cluster analysis. Psychometrika, 46, no. 2:187–199, 1981.

        Returns:

        """
        SW = self.execute_subprocess('sum_intra_cluster_distances')
        SW = sum(SW.values())
        SB = self.execute_subprocess('sum_inter_cluster_distances')
        SB = sum(SB.values())

        n_w = self.execute_subprocess('intra_cluster_distances')
        n_w = sum([(j.shape[0] * (j.shape[0] - 1)) / 2 for i, j in n_w.items()])
        n_w = int(n_w)

        n_t = (self.X.shape[0] * (self.X.shape[0] - 1)) / 2

        n_b = n_t - n_w

        return (SW / n_w - SB / n_b) * (math.sqrt(n_w * n_b) / n_t)

    def ratkowsky_lance(self):
        BG = self.execute_subprocess('between_group_scatter_matrix')

        TS = self.execute_subprocess('total_scatter_matrix')
        c_bar_squared = np.mean(np.diag(BG) / np.diag(TS))
        return math.sqrt(c_bar_squared / len(np.unique(self.y)))

    def ray_turi(self):
        """
        - Ray et al. Determination of number of clusters in k-means clustering and application in colour image
        segmentation. 4th International Conference on Advances in Pattern Recognition and Digital Techniques, 1999.
        = (WGSS/N)/ (min(||Gk - Gk'||)**2)
        Returns:

        """
        numerator = self.execute_subprocess('within_group_dispersion')
        numerator = numerator / self.X.shape[0]
        denominator = self.execute_subprocess('pairwise_cluster_centers_distances')
        denominator = min(denominator.values()) ** 2
        return numerator / denominator

    def rubin(self):
        T = self.execute_subprocess('total_scatter_matrix')
        W = self.execute_subprocess('within_group_scatter_matrix')
        return np.linalg.norm(T) / np.linalg.norm(W)

    def scott_symons(self):
        wgk = self.execute_subprocess('within_group_scatter_matrices')

        ss = {i: len(self.y[self.y == i]) * np.log(np.linalg.det(j / len(self.y[self.y == i]))) for i, j in wgk.items()}
        return sum(ss.values())

    def sd_dis(self):
        """
        Halkidi et al. On clustering validation techniques. J. Intell. Inf. Syst., 2001.
        =(Dmax/Dmin) * Σ(Σ(1/|Gk-Gk'|))
        Dmax: the maximum distance of cluster centers
        Dmin: the minimum distance of cluster centers
        Returns:
            float: the value of SD_dis (dispersion)
        """

        ccenters_distances = self.execute_subprocess('pairwise_cluster_centers_distances')
        Dmax = max(ccenters_distances.values())
        Dmin = min(ccenters_distances.values())

        sums = {}
        for (key1, key2), value in ccenters_distances.items():
            if key1 in sums:
                sums[key1] += value
            else:
                sums[key1] = value
            if key2 in sums:
                sums[key2] += value
            else:
                sums[key2] = value
        sums = sum([1 / x for x in sums.values()])

        return (Dmax / Dmin) * sums

    def sd_scat(self):
        """
        Halkidi et al. On clustering validation techniques. J. Intell. Inf. Syst., 2001.
        Returns:

        """
        total_scatter_matrix = self.execute_subprocess('total_scatter_matrix')
        total_scatter_matrix = total_scatter_matrix / self.X.shape[0]
        variances = np.diag(total_scatter_matrix)

        wg_scatter_matrices = self.execute_subprocess('within_group_scatter_matrices')
        wg_scatter_matrices = {i: j / len(self.X[self.y == i]) for i, j in wg_scatter_matrices.items()}
        sum_of_wg_norms = sum([np.linalg.norm(np.diag(x)) for x in wg_scatter_matrices.values()])

        sd_scat = (sum_of_wg_norms / len(np.unique(self.y))) / np.linalg.norm(variances)
        return sd_scat

    def sdbw(self):
        self.calculate_icvi(['sd_scat'])
        S = self.cvi_results['sd_scat']

        # calculate sigma
        # wg_scatter_matrices = self.execute_subprocess('within_group_scatter_matrices')
        # wg_scatter_matrices = {i: j / len(self.X[self.y == i]) for i, j in wg_scatter_matrices.items()}
        # sigma = sum([np.linalg.norm(np.diag(x)) for x in wg_scatter_matrices.values()])
        # sigma = (1 / len(np.unique(self.y))) * math.sqrt(sigma)

        cluster_stds = []
        for c in np.unique(self.y):
            std = np.std(self.X[self.y == c])  # Standard deviation per dimension
            cluster_stds.append(np.mean(std))

        sigma = np.mean(cluster_stds)


        ccenters = self.execute_subprocess('cluster_centers')
        midpoints = {(i, j): find_midpoint(ccenters[i].reshape(1, -1), ccenters[j].reshape(1, -1)) for i, j in
                     combinations(ccenters.keys(), r=2)}
        midpoint_distances = {i: cdist(midpoints[i], np.concatenate([self.X[self.y == i[0]], self.X[self.y == i[1]]]))
                              for i, j in midpoints.items()}
        Hkk = {i: np.sum(j < sigma) for i, j in midpoint_distances.items()}

        gk = self.execute_subprocess('distances_from_cluster_centers')
        gk_ = self.execute_subprocess('distances_from_other_cluster_centers')

        sum_ = 0
        for comb in combinations(ccenters.keys(), r=2):
            k = comb[0]
            k_ = comb[1]
            if k_ > k:
                k_ += -1

            gkk_ = np.sum(gk[k] < sigma) + np.sum(gk_[k][:, k_] < sigma)

            k = comb[0]
            k_ = comb[1]
            if k_ < k:
                k += -1
            gk_k = np.sum(gk[k_] < sigma) + np.sum(gk_[k_][:, k] < sigma)

            sum_ += Hkk[comb] / max(gk_k, gkk_)

        return (sum_ * (2 / (self.noclusters * (self.noclusters - 1)))) + S

    def tau(self):
        s_plus, s_minus, nb, nw = self.execute_subprocess('s_values')
        return (s_plus - s_minus) / (nb * nw * ((self.X.shape[0] * (self.X.shape[0] - 1)) / 2)) * 1 / 2

    def trace_w(self):
        WG = self.execute_subprocess('within_group_scatter_matrix')
        return np.trace(WG)

    def trace_wib(self):
        WG = self.execute_subprocess('within_group_scatter_matrix')
        BG = self.execute_subprocess('between_group_scatter_matrix')
        return np.trace(np.linalg.inv(WG) * BG)

    def wemmert_gancarski(self):
        """
        =
        Returns:

        """
        dist_from_ccenter = self.execute_subprocess('distances_from_cluster_centers')
        dist_from_other_ccenters = self.execute_subprocess('distances_from_other_cluster_centers')
        dist_from_other_ccenters = {i: np.min(j, axis=1) for i, j in dist_from_other_ccenters.items()}
        new = {}
        for key in dist_from_ccenter:
            new[key] = dist_from_ccenter[key].reshape(1, -1) / dist_from_other_ccenters[key]

        new = {i: np.mean(j) for i, j in new.items()}

        new_ = {j: (1 - x if x < 1 else 0) for j, x in new.items()}
        new_ = {i: j * len(self.y[self.y == i]) for i, j in new_.items()}
        return sum(new_.values()) / self.X.shape[0]

    def xie_beni(self):

        """
        - X.L. Xie and G. Beni. A validity measure for fuzzy clustering. IEEE Transactions on Pattern Analysis and
        Machine Intelligence, 1991.
        = 1/N WGSS/min (min(d(Cik, Cjk')))**2,
        where d(Cik, Cjk) is the pairwise inter cluster distances
        Returns:

        """
        numerator = self.execute_subprocess('within_group_dispersion')
        denominator = self.execute_subprocess('min_inter_cluster_distances')
        denominator = min(denominator.values())
        return (1 / self.X.shape[0]) * (numerator / denominator ** 2)
