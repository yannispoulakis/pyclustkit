from json import JSONDecodeError

from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, AffinityPropagation, SpectralClustering
import numpy as np
import json
from itertools import product
import gradio as gr
from collections import Counter
import matplotlib.pyplot as plt
import os
from pyclustkit.eval import CVIToolbox
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
from pathlib import Path


cvi_list = list(CVIToolbox(np.array([1, 2]), np.array([1, 2])).cvi_methods_list.keys())
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99'] #used in plots
best_idx_type = CVIToolbox.cvi_opt_type

search_space = """
                {
                    "KMeans": {"n_clusters": [2, 12, 1], 
                        "algorithm": ["lloyd", "elkan"],
                        "max_iter": 500, 
                        "init": ["k-means++", "random"]}, 
                                                  
                    "DBSCAN": { 
                        "eps": [0.01, 1, 0.05], 
                        "min_samples": [2, 21,1], 
                        "metric": ["euclidean", "l1", "cosine"]}, 
                                                
                    "Agglomerative": {
                        "n_clusters": [2, 12, 1], 
                        "metric": ["euclidean", "l1", "cosine"], 
                        "linkage": ["ward", "complete", "average", "single"]}, 
                                    
                    "Affinity Propagation": {
                        "damping": [0.5, 1, 0.1]}, 
                                    
                    "Spectral Clustering": {
                        "n_clusters": [2, 12, 1], 
                        "gamma": [0.8, 1.2, 0.1],
                        "affinity": ["nearest_neighbors", "rbf"], 
                        "n_neighbors": [2, 10, 1],
                        "assign_labels": ["kmeans", "discretize", "cluster_qr"]}   
                                }
                                """

def transform_search_space(json_input):
    """
    Utility function that takes a dictionary and expands the values.
     The expected dict format should be as follows:
        {Alg_1: {par_1: [start, end, step], ....}, ....}

    (a) list of floats = np.arrange(l[0], l[1], l[2])
    (b) list of ints = range(l[0], l[1], l[2])
    (c) single var = [var]
    Args:
        json_input (dict): The dictionary that describes parametric spaces
                            example: {"KMeans": {"no_clusters": [], ......}}

    Returns:
        dict: The updated search space
    """
    for key in json_input:
        for key_ in json_input[key]:
            if type(json_input[key][key_]) is list and type(json_input[key][key_][0]) is int:
                json_input[key][key_] = list(
                    range(json_input[key][key_][0], json_input[key][key_][1], json_input[key][key_][2]))

            elif type(json_input[key][key_]) is list and type(json_input[key][key_][0]) is float:
                json_input[key][key_] = list(
                    np.arange(json_input[key][key_][0], json_input[key][key_][1], json_input[key][key_][2]))

            elif type(json_input[key][key_]) is not list:
                json_input[key][key_] = [json_input[key][key_]]
    return json_input

def calculate_cvi(x, y,  custom_set, cvi_search_type="all"):
    """
    Calculates cvi with pyclustkit.eval
    Args:
        x (np.ndarray or pd.DataFrame): The dataset to use
        y (np.ndarray or pd.DataFrame): The clustering labels as found by any clustering algorithm
        cvi_search_type (str): This parameter should be 'all' or it will be ignored to calculate a custom set given.
        custom_set (list): The list of CVI to calculate.

    Returns:
        dict: The CVI calculated
    """
    x = np.array(x)
    y = np.array(y)

    if cvi_search_type.lower() == 'all':
        cvit = CVIToolbox(x, y)
        cvit.calculate_icvi(cvi="all")
    else:
        cvi = custom_set
        cvit = CVIToolbox(x, y)
        cvit.calculate_icvi(cvi=cvi)

    return cvit.cvi_results

def find_best_per_cvi(data_id):
    """
    Finds the best configuration according to each CVI from given trial results
    Args:
        results_dict (dict): Trial results, referenced as master_results in main.py
        data_id (str): The dataset to search the best configurations for

    Returns:
        dict: The best configuration for each CVI
    """


    # Get all trial results in a list
    if not os.path.exists(Path(f"results/_es.json")):
        gr.Info("Attempted to find Best Configuration according to CVI , but no ES results were found in the results "
                "folder.")
        return
    with open(Path(f"results/_es.json"), "r") as f:
        es_results = json.load(f)
    all_configs = []
    for key in es_results:
        all_configs += es_results[key]

    best_config_per_cvi = {}
    for cvi in cvi_list:
        try:
            if cvi in best_idx_type["max"]:
                best_config_per_cvi[cvi] = max(all_configs, key=lambda x: x["cvi"][cvi])

            elif cvi in best_idx_type["min"]:
                best_config_per_cvi[cvi] = min(all_configs, key=lambda x: x["cvi"][cvi])

            elif cvi in best_idx_type["max_diff"]:
                sorted_configs = sorted(all_configs, key=lambda x: x["cvi"][cvi])
                diffs = [sorted_configs[i + 1]["cvi"][cvi] -
                             sorted_configs[i]["cvi"][cvi] for i in range(len(sorted_configs) - 1)]
                max_diff_index = diffs.index(max(diffs))
                best_config_per_cvi[cvi] = sorted_configs[max_diff_index + 1]

            elif cvi in best_idx_type["min_diff"]:
                sorted_configs = sorted(all_configs, key=lambda x: x["cvi"][cvi])
                diffs = [sorted_configs[i + 1]["cvi"][cvi] -
                             sorted_configs[i]["cvi"][cvi] for i in range(len(sorted_configs) - 1)]
                max_diff_index = diffs.index(min(diffs))
                best_config_per_cvi[cvi] = sorted_configs[max_diff_index + 1]

        except Exception as e:
            print(f"Eror in finding best config: {cvi}")
            print(e)
            print(f"Eror in finding best config: {cvi}")
            continue

    print(f"Found {len(best_config_per_cvi.keys())} best configs")
    print(best_config_per_cvi)
    return best_config_per_cvi

def create_plots_from_es(best_config_per_cvi):
    """
    Creates two plots to serve in the UI:
    (a) histogram that displays the frequency of value for the number of clusters parameters as found per the best
        trials for each CVI.
    (b) Pie Chart used to show the frequency of the best algorithm found for each CVI.

    WARNING: This function returns None.
            Plots are saved.
    Args:
        best_config_per_cvi (dict): A dictionary that contains the trial information.
                                    Should be in the form {'algorithm': str, 'params': {}, 'labels': [], 'cvi': {}}

    Returns:
        None
    """
    print("\033[92m Creating plots from exhaustive search....")
    best_alg_count = []
    no_clusters_found_per_best = []

    for key in best_config_per_cvi:
        best_alg_count.append(best_config_per_cvi[key]["algorithm"])
        no_clusters_found_per_best.append(len(set(best_config_per_cvi[key]["labels"])))
    counter = Counter(best_alg_count)

    # ---> First Plot: Pie Chart (Number of times (per CVI) each algorithm was appointed "Best"
    pie_labels = list(counter.keys())
    pie_values = list(counter.values())

    plt.figure(figsize=(6, 6))
    plt.title("Best Algorithm Count Per CVI", fontsize=14, fontweight='bold')

    wedges, texts, autotexts = plt.pie(
        pie_values,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,  # Apply custom colors
        shadow=True,  # Add shadow for a 3D effect
        wedgeprops={'edgecolor': 'black'}  # Add a border around the slices
    )

    plt.legend(wedges, pie_labels,loc="lower right")
    plt.axis('equal')

    plt.savefig("best_alg_pie.png", dpi=300)  # Increase dpi for better resolution
    plt.close()

    # ---> Second Plot: Histogram
    plt.figure(figsize=(6, 6))

    # Histogram with enhanced styling
    plt.hist(
        no_clusters_found_per_best,
        bins=range(2, 22),  # Custom bin range
        edgecolor='black',  # Edge color for clarity
        color='#66b3ff',  # Custom bar color for aesthetics
        alpha=0.8  # Slight transparency for the bars
    )

    # Customize the ticks and grid
    plt.xticks(range(0, 22), fontsize=10)  # Set x-tick labels and size
    plt.yticks(fontsize=10)  # Set y-tick labels and size
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add a light grid for y-axis

    # Add labels and title with increased font sizes and bold titles
    plt.title("No Clusters Found In Best Configuration Per CVI", fontsize=14, fontweight='bold')
    plt.xlabel("No Clusters", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)

    # Save and close the figure with a higher resolution
    plt.savefig("no_clusters_hist.png", dpi=300)  # Save with high resolution for clarity
    plt.close()
    print("\033[92m Plots created succesfully!")


def exhaustive_search( data_id, df, json_input, operations_state):
    """

    Args:
        master_results (dict): The dictionary that contains trial data per dataset
                                {'data id': {'alg1': [], alg2: []...}}
        data_id (str): The assigned data ID for the dataset currently active
        df (pd.DataFrame): The data provided/generated
        json_input (str): The search space defined as a string of a dict

    Returns:
        tuple: A tuple that contains visibility and content updates for Gradio components.
            (1) master_results - Content
            (2) ES success message - Content
            (3) Download Results button - Visibility
            (4) Pie-chart Image - Content
            (5) Histogram Image - Content
            (6) Best Config per CVI - Content

    Raises:
        JSONDecodeError: When json that defines the parametric space is invalid. Additionally throws and error in UI.
    """
    clustering_methods = {"KMeans": KMeans, "DBSCAN": DBSCAN, "Agglomerative": AgglomerativeClustering,
                          "Affinity Propagation": AffinityPropagation, "Spectral Clustering": SpectralClustering}

    try:
        json_input = json.loads(json_input)
        print(json_input)
    except JSONDecodeError :
        raise gr.Error("Invalid JSON in parameter space definition")
    except Exception as e:
        raise gr.Error(str(e))


    json_input = transform_search_space(json_input)

    param_combinations_per_alg = {}
    master = dict([(x,[]) for x in json_input.keys()])
    for key_alg in json_input:
        print(f"-----------------------({key_alg})--------------------------------------------------")
        # Define and iterate over parameter space for each algorithm
        param_combinations_per_alg[key_alg] = list(product(*list(json_input[key_alg].values())))
        gr.Info(f"Grid Search: {key_alg} : {sum([len(param_combinations_per_alg[x]) for x in 
                                                  param_combinations_per_alg])} total models")

        for param_combination in param_combinations_per_alg[key_alg]:
            print(f"-----------------------({param_combination})--------------------------------------------------")
            trial_values = {}
            params = dict(zip(json_input[key_alg].keys(), list(param_combination)))
            try:
                labels_ = clustering_methods[key_alg](**params).fit_predict(df)
            except ValueError:
                continue
            print(f"labels ok")
            if len(set(labels_)) == 1:
                continue
            else:
                # idx_custom_set is only relevant if idx_search_type != "all", otherwise it is ignored
                cvi = calculate_cvi(x=df, y=labels_, custom_set=[1,2])
                print(f"CVI ok")
                trial_values["algorithm"] = key_alg
                trial_values["params"] = params
                trial_values["labels"] = list([int(x) for x in list(labels_)])
                trial_values["cvi"] = cvi

                master[key_alg].append(trial_values)



    # Save Results
    with open(os.path.join(os.getcwd(), "results", f"_es.json"), "w") as f:
            json.dump(master, f)

    best_config_per_cvi = find_best_per_cvi( data_id)
    create_plots_from_es(best_config_per_cvi)

    operations_state["configurations-search"] = True
    return ( gr.update(visible=True, value="<h2 style= text-align:center;>ES success</h2>"),
                gr.update(visible=True), "best_alg_pie.png",
                "no_clusters_hist.png", best_config_per_cvi,
                operations_state)

# -------------------------------------------------Tab 2-------------------------

def serve_clustering_visualizations(best_config_per_cvi, df_reduced, cvi, reduction_method, df):
    data = np.array(df_reduced[reduction_method])
    config = best_config_per_cvi[cvi]
    labels = config["labels"]

    cmap = plt.cm.get_cmap("viridis", len(np.unique(labels)))
    norm = mcolors.BoundaryNorm(np.arange(-0.5, len(np.unique(labels)), 1), cmap.N)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap, norm=norm, s=50)
    plt.colorbar(scatter, ticks=np.unique(labels))
    plt.title("Dataset Scatterplot}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.savefig("clusters_visualized.png")
    fig1 = plt.gcf()

    return fig1

def create_scatterplot(data, labels):
    data = np.array(data)
    cmap = plt.cm.get_cmap("viridis", len(np.unique(labels)))
    norm = mcolors.BoundaryNorm(np.arange(-0.5, len(np.unique(labels)), 1), cmap.N)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap, norm=norm, s=50)
    plt.colorbar(scatter, ticks=np.unique(labels))
    plt.title("Dataset Scatterplot")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.savefig("clusters_visualized.png")
    fig1 = plt.gcf()

    return fig1

def on_best_cvi_change(cvi, best_config_per_cvi, df_reduced, df , df_needs_reduction, df_reduction_method):
    """
    This method is applied when the index is changed in tab-2/Clustering Exploration
        (1) It displays the full configuration found when optimizing the index
        (2) -If data need reduction (>2 columns)- Performs selected dimensionality reduction and updates repository.
        (3) Creates and returns plots for visualization

    Args:
        cvi:
        best_config_per_cvi:
        df_reduced:
        df:
        df_needs_reduction:
        df_reduction_method:

    Returns:

    """


    methods = {"T-SNE": TSNE, "PCA": PCA, "MDS": MDS}

    # --- (1) ---
    try:
        cache_best_config = best_config_per_cvi[cvi]
    except KeyError:
        raise gr.Error(f"CVI: {cvi} was not calculated during model search.")
    best_config = f"Algorithm: {best_config_per_cvi[cvi]['algorithm']}\nParameters: {best_config_per_cvi[cvi]['params']}"

    # --- (2) ---
    if df_needs_reduction:
        if df_reduced[df_reduction_method] is None:
            gr.Info("First time applying this dimensionality reduction method, it may take a while for big datasets.")
            df_reduced[df_reduction_method] = methods[df_reduction_method](n_components=2).fit_transform(df)
            data = df_reduced[df_reduction_method]
        else:
            data = df_reduced[df_reduction_method]
    else:
        data = df

    scatter_plot = create_scatterplot(data, cache_best_config["labels"])
    return best_config, df_reduced, scatter_plot



        #data = df_reduced[df_reduction_method]
    #else:
      #  data = df