from ..metalearning.descriptive import *
from ..metalearning.statistics import *
from ..metalearning.t_squared import *
from ..metalearning.landmark import CVIMF
from ..metalearning.cad import CaD
from ..metalearning.marcoge_extractor import marcoge_mf

import traceback


class MFExtractor:
    def __init__(self, data=None):
        self.df = np.array(data)
        self.cvimf = CVIMF()
        self.cad = CaD()
        self.mf_categories = ["landmark", "statistics", "descriptive", "similarity-vector"]
        self.mf_papers = ["Ferrari", "Souto", "Nascimento", "Vukicevic", "AutoClust", "TPE-AutoClust", 'Marco-GE',
                          "AutoCluster", "cSmartML", "Pimentel"]
        self.errors= []
        self.meta_features = {
            # No instances & No attributes
            "log2_no_instances": {"value": None, "category": "Descriptive", "included_in":
                ["Ferrari"], "method": log2_instances},
            "log2_no_attributes": {"value": None, "category": "Descriptive", "included_in":
                ["Ferrari"], "method": log2_attributes},
            "log10_no_instances": {"value": None, "category": "Descriptive", "included_in":
                ["Souto", "Nascimento", "Vukicevic", "AutoCluster"],
                                   "method": log10_instances},
            "log10_no_attributes": {"value": None, "category": "Descriptive", "included_in":
                ["Souto", "Nascimento", "Vukicevic", "AutoCluster"],
                                    "method": log10_attributes},

            "instances_to_features": {"value": None, "category": "Descriptive", "included_in":
                [], "method": instances_to_features_ratio},
            "log_instances_to_features": {"value": None, "category": "Descriptive", "included_in":
                ["Souto", "Nascimento", "Vukicevic", "AutoCluster"],
                                          "method":log_instances_to_features_ratio},
            "features_to_instances": {"value": None, "category": "Descriptive", "included_in":
                [], "method": features_to_instances_ratio},
            "log_features_to_instances": {"value": None, "category": "Descriptive", "included_in":
                [], "method": log_features_to_instances_ratio},

            "Hopkin's statistic": {"value": None, "category": "Statistics", "included_in":
                ["AutoCluster"], "method": hopkins},

            "pct_of_discrete_attributes": {"value": None, "category": "Descriptive", "included_in":
                ["Ferrari"], "method": pct_of_discrete},
            "mean_entropy_of_discrete_attributes": {"value": None, "category": "Statistics", "included_in":
                ["Ferrari"], "method": mean_entropy_of_discrete},

            "mean_absolute_correlation_of_continuous_attributes":
                {"value": None, "category": "Statistics", "included_in": ["Ferrari"],
                 "method": mean_absolute_continuous_feature_correlation},
            "mean_concentration_of_discrete_attributes": {"value": None, "category": "Statistics", "included_in":
                ["Ferrari"], "method": mean_concentration_between_discrete},

            "pca_95_components_to_features_ratio":
                {"value": None, "category": "Statistics", "included_in": ["AutoCluster"],
                 "method": pca_95_deviations_to_features_ratio},
            "pca_first_component_skewness": {"value": None, "category": "Statistics", "included_in":
                ["AutoCluster"], "method": skewness_of_pca_first_component},
            "pca_first_component_kurtosis": {"value": None, "category": "Statistics", "included_in":
                ["AutoCluster"], "method": kurtosis_of_pca_first_component},

            "pct_of_outliers": {"value": None, "category": "Statistics", "included_in":
                ["Souto", "Nascimento", "Vukicevic", "Ferrari"],
                                "method": pct_of_outliers},
            "pct_of_mv": {"value": None, "category": "Statistics", "included_in":
                ["Souto", "Nascimento", "Vukicevic"],
                          "method": pct_of_mv},

            # skewness vector of continuous attributes
            "min_skewness_of_continuous_attributes": {"value": None, "category": "Statistics", "included_in":
                ["AutoCluster"], "method": min_skewness_of_continuous},
            "max_skewness_of_continuous_attributes": {"value": None, "category": "Statistics", "included_in":
                ["AutoCluster"], "method": max_skewness_of_continuous},
            "mean_skewness_of_continuous_attributes": {"value": None, "category": "Statistics", "included_in":
                ["Ferrari", "AutoCluster"], "method": mean_skewness_of_continuous},
            "std_skewness_of_continuous_attributes": {"value": None, "category": "Statistics", "included_in":
                ["AutoCluster"], "method": std_skewness_of_continuous},

            # Kurtosis vector of continuous attributes
            "min_kurtosis_of_continuous_attributes": {"value": None, "category": "Statistics", "included_in":
                ["AutoCluster"], "method": min_kurtosis_of_continuous},
            "max_kurtosis_of_continuous_attributes": {"value": None, "category": "Statistics", "included_in":
                ["AutoCluster"], "method": max_kurtosis_of_continuous},
            "mean_kurtosis_of_continuous_attributes": {"value": None, "category": "Statistics", "included_in":
                ["Ferrari", "AutoCluster"], "method": mean_kurtosis_of_continuous},
            "std_kurtosis_of_continuous_attributes": {"value": None, "category": "Statistics", "included_in":
                ["AutoCluster"], "method": std_kurtosis_of_continuous},

            # t-squared
            "t_squared_mv_normality": {"value": None, "category": "Statistics", "included_in":
                ["Souto", "Nascimento", "Vukicevic"],
                                       "method": mv_normality_test},
            "t_squared_skewness": {"value": None, "category": "Statistics", "included_in":
                ["Souto", "Nascimento", "Vukicevic"],
                                   "method": t_squared_skewness},
            "t_squared_outliers": {"value": None, "category": "Statistics", "included_in":
                ["Souto", "Nascimento", "Vukicevic"],
                                   "method": t_squared_outliers},
            # Landmark
            # (a) CVI
            "meanshift_silhouette": {"value": None, "category": "Landmark",
                                     "included_in": ["AutoClust", "TPE-AutoClust", "cSmartML"],
                                     "method": self.cvimf.calculate_cvi, "method_additional_parameters":
                                         {"algorithm": "meanshift", "cvi_name": "silhouette"}},
            "meanshift_dunn": {"value": None, "category": "Landmark", "included_in": ["AutoClust"],
                               "method": self.cvimf.calculate_cvi, "method_additional_parameters":
                                   {"algorithm": "meanshift", "cvi_name": "dunn"}},
            "meanshift_cindex": {"value": None, "category": "Landmark", "included_in": ["AutoClust"],
                                 "method": self.cvimf.calculate_cvi, "method_additional_parameters":
                                     {"algorithm": "meanshift", "cvi_name": "c_index"}},
            "meanshift_calinski_harabasz": {"value": None, "category": "Landmark",
                                            "included_in": ["AutoClust", "TPE-AutoClust", "cSmartML"],
                                            "method": self.cvimf.calculate_cvi, "method_additional_parameters":
                                                {"algorithm": "meanshift", "cvi_name": "calinski_harabasz"}},
            "meanshift_davies_bouldin": {"value": None, "category": "Landmark",
                                         "included_in": ["AutoClust", "TPE-AutoClust", "cSmartML"],
                                         "method": self.cvimf.calculate_cvi, "method_additional_parameters":
                                             {"algorithm": "meanshift", "cvi_name": "davies_bouldin"}},
            "meanshift_sdbw": {"value": None, "category": "Landmark", "included_in": ["AutoClust"],
                               "method": self.cvimf.calculate_cvi, "method_additional_parameters":
                                   {"algorithm": "meanshift", "cvi_name": "s_dbw"}},
            "meanshift_cdbw": {"value": None, "category": "Landmark", "included_in": ["AutoClust"],
                               "method": self.cvimf.calculate_cvi, "method_additional_parameters":
                                   {"algorithm": "meanshift", "cvi_name": "cdbw"}},
            "meanshift_tau": {"value": None, "category": "Landmark", "included_in": ["AutoClust"],
                              "method": self.cvimf.calculate_cvi, "method_additional_parameters":
                                  {"algorithm": "meanshift", "cvi_name": "tau"}},
            "meanshift_ratkowsky_lance": {"value": None, "category": "Landmark", "included_in": ["AutoClust"],
                                          "method": self.cvimf.calculate_cvi, "method_additional_parameters":
                                              {"algorithm": "meanshift", "cvi_name": "ratkowsky_lance"}},
            "meanshift_mcclain_rao": {"value": None, "category": "Landmark", "included_in": ["AutoClust"],
                                      "method": self.cvimf.calculate_cvi, "method_additional_parameters":
                                          {"algorithm": "meanshift", "cvi_name": "mcclain_rao"}},

            "DBSCAN_silhouette": {"value": None, "category": "Landmark", "included_in": ["TPE-AutoClust", "cSmartML"],
                                  "method": self.cvimf.calculate_cvi, "method_additional_parameters":
                                      {"algorithm": "dbscan", "cvi_name": "silhouette"}},
            "DBSCAN_davies_bouldin": {"value": None, "category": "Landmark",
                                      "included_in": ["TPE-AutoClust", "cSmartML"],
                                      "method": self.cvimf.calculate_cvi, "method_additional_parameters":
                                          {"algorithm": "dbscan", "cvi_name": "davies_bouldin"}},
            "DBSCAN_calinski_harabasz": {"value": None, "category": "Landmark",
                                         "included_in": ["TPE-AutoClust", "cSmartML"],
                                         "method": self.cvimf.calculate_cvi, "method_additional_parameters":
                                             {"algorithm": "dbscan", "cvi_name": "calinski_harabasz"}},
            "OPTICS_silhouette": {"value": None, "category": "Landmark", "included_in": ["TPE-AutoClust", "cSmartML"],
                                  "method": self.cvimf.calculate_cvi, "method_additional_parameters":
                                      {"algorithm": "optics", "cvi_name": "silhouette"}},
            "OPTICS_davies_bouldin": {"value": None, "category": "Landmark",
                                      "included_in": ["TPE-AutoClust", "cSmartML"],
                                      "method": self.cvimf.calculate_cvi, "method_additional_parameters":
                                          {"algorithm": "optics", "cvi_name": "davies_bouldin"}},
            "OPTICS_calinski_harabasz": {"value": None, "category": "Landmark",
                                         "included_in": ["TPE-AutoClust", "cSmartML"],
                                         "method": self.cvimf.calculate_cvi, "method_additional_parameters":
                                             {"algorithm": "optics", "cvi_name": "calinski_harabasz"}},
            "ferrari_distance_based": {"value": None, "category": "similarity-vector",
                                       "included_in": ["Ferrari", "Pimentel", "TPE-AutoClust", "CSmartML"],
                                       "method": self.cad.ferrari_distance, "method_additional_parameters":
                                           {}},
            "correlation_based": {"value": None, "category": "similarity-vector",
                                  "included_in": [ "Pimentel"],
                                  "method": self.cad.correlation, "method_additional_parameters":
                                      {}},
            "correlation_and_distance": {"value": None, "category": "similarity-vector",
                                         "included_in": [ "Pimentel"],
                                         "method": self.cad.cad, "method_additional_parameters":
                                             {}},
            "marcoge": {"value": None, "category": "graph-based",
                        "included_in": ["Marco-GE"],
                        "method": marcoge_mf, "method_additional_parameters":
                            {}},
        }

    def search_mf(self, name=None, category=None, included_in=None, search_type="full_search"):
        """
        Finds a subset of supported meta-features and returns their attributes (method, category, value (if computed),
        etc.)
        :param name: Used to find the details of a specific CVI by name.
        :type name: str
        :param category: The category of meta-features to find. Available categories are found with .mf_categories .
        :type category: str
        :param included_in: Find meta-features per work. Available works are found with .mf_papers .
        :type included_in: str
        :param search_type: Return content. Valid values are full_search, names, values.
        :type search_type: str
        :return: Returns the subset of meta-features of interest defined by the other input parameters.
        :rtype: dict
        """
        assert search_type in ["full_search", "values", "names"], ("parameter: search type must be one of [full_search,"
                                                                   "values, names]")
        if category is not None:
            assert category.lower() in self.mf_categories, f"parameter: category must be one of {self.mf_categories}"

        if name is not None:
            mf = self.meta_features[name]

        elif category is not None:
            mf = dict([(x, self.meta_features[x]) for x in self.meta_features if
                       category.lower() == self.meta_features[x]["category"].lower()])

        elif included_in is not None:
            mf = dict([(x, self.meta_features[x]) for x in self.meta_features if
                       included_in.lower() in [y.lower() for y in self.meta_features[x]["included_in"]]])
        else:
            mf = self.meta_features

        if search_type == "names":
            return list(mf.keys())
        elif search_type == "values":
            return dict([(x, mf[x]["value"]) for x in mf])
        else:
            return mf

    def calculate_mf(self, name=None, category=None, included_in=None):
        mf_ = self.search_mf(name=name, category=category, included_in=included_in)
        for mf in mf_:
            mf_value = None
            print("-----------")
            print(f'calculating meta-feature: {mf}')
            if 'method_additional_parameters' in mf_[mf].keys():
                extra_method_parameters = mf_[mf]['method_additional_parameters']
            else:
                extra_method_parameters = {}
            try:
                mf_value = mf_[mf]['method'](self.df, **extra_method_parameters)
                print(mf_value)
            except Exception as e:
                print("-------------------")
                print(e)
                print(traceback.format_exc())
                print("----------------------")
                self.errors.append((mf, e))
            self.meta_features[mf]["value"] = mf_value

mfe = MFExtractor()
len(mfe.meta_features.keys())