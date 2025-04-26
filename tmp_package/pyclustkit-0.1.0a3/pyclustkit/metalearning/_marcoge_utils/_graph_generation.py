# TODO: CHANGE TO KEEP ONLY TEMP FILES
# TODO: Fix bug in  edge sparse matrix creation
import os
from traceback import format_list

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix

import dgl
import torch
from ...deepwalk_custom.core import process_graph


core_dir = os.path.dirname(os.path.dirname(__file__))

def extract_edges(similarity_matrix, dataset_name="", threshold=0.9):
    """
    This function finds the edges to a graph formed by a dataset's similarity matrix. It only forms edges that the
     corresponding cosine similarity distance is below a threshold. Edgelist is also written to a file.

    :param dataset_name:
    :type dataset_name:
    :param similarity_matrix: A NxN matrix with dataset's instance pairwise distances.
    :type similarity_matrix: np.ndarray
    :param threshold: Values in the similarity matrix lower than the threshold will be removed.
    :type threshold: float
    :return: Sparse array that contains edges
    :rtype: scipy.sparse.coo_matrix
    """

    edge_list = []
    src = []
    dst = []
    weight = []
    similarity_matrix[similarity_matrix < threshold] = 0
    for i in range(similarity_matrix.shape[0]):
        if np.count_nonzero(similarity_matrix[i]) == 1:
            similarity_matrix_ = np.delete(similarity_matrix, i,0)
            similarity_matrix_ = np.delete(similarity_matrix_, i, 1)

    for i in range(similarity_matrix.shape[0]):
        for c in range(i + 1, similarity_matrix.shape[1]):
            val = similarity_matrix[i, c]
            if val != 0:
                if i == 34 or c == 34:
                    pass
                    # print("asdaslkdjaldfkj:LSdkgjhs;lkgjhs;ldkgj;sldkjg")
                d = 1
                edge_list.append([" ".join([str(i), str(c), str(val)]) + "\n"])
                src.append(i)
                dst.append(c)
                weight.append(val)
    if len(edge_list) == 0:
        print('No edges formed, consider adjusting threshold.')
        return None

    if not os.path.exists(os.path.join(core_dir, "temp", "edges")):
        os.mkdir(os.path.join(core_dir, "temp", "edges"))

    path_to_write = os.path.join(core_dir, "temp", "edges", f"el_{dataset_name}.file")
    with open(path_to_write, 'w') as f:
        for lis in edge_list:
            f.writelines(lis)

    src_arr = np.array(src)
    dst_arr = np.array(dst)
    weight_arr = np.array(weight)
    total_nodes = len(set(dst + src))
    # print(total_nodes)
    # print(f"--------------------{weight_arr.shape}")
    # print(max(src+dst))
    # print(len(np.unique((src + dst))))
    # print(np.unique(src+dst))
    # print(set(np.unique((src + dst))).difference( set(list(range(0,106)))))
    src_arr = np.add(src_arr,1)
    weight_arr = np.add(weight_arr, 1)
    print(max(src_arr))
    print(max(dst_arr))

    return coo_matrix((weight_arr, (src_arr, dst_arr)), (total_nodes, total_nodes))


def extract_node_embeddings(dataset_name):

    """
    Extracts node embeddings for a given graph.
    :param dataset_name:
    :type dataset_name:
    :return:
    :rtype:
    """
    print("Starting Node embeddings extraction (DeepWalk)...... ")
    if not os.path.exists(os.path.join(core_dir, "temp", "embeddings")):
        os.mkdir(os.path.join(core_dir, "temp", "embeddings"))

    edge_list_path = os.path.join(core_dir, "temp", "edges", f"el_{dataset_name}.file")
    path_out = os.path.join(core_dir, "temp", "embeddings", f"ne_{dataset_name}.file")

    process_graph(input_file=edge_list_path,output_file=path_out,format="weighted_edgelist")

    pass


def graph_implementation(edge_list_coo,
                         dataset_name):
    graph = dgl.from_scipy(edge_list_coo, eweight_name='weight')
    graph = dgl.add_reverse_edges(graph, copy_edata=True)
    graph = dgl.add_self_loop(graph)
    if '_ID' in graph.edata:
        graph.edata.pop('_ID')

    node_emb_path = os.path.join(core_dir, "temp", "embeddings", f"ne_{dataset_name}.file")
    with open(node_emb_path, 'r') as f:

        line = f.readline()
        line = f.readline()
        node_feat_dict = {}
        while line:
            cur_line = line.split(" ")
            node_feat_dict[int(cur_line[0])] = [float(val) for val in cur_line[1:]]
            line = f.readline()

    sorted_node_feat_dict = dict(sorted(node_feat_dict.items(), key=lambda e: e[0]))
    node_feat_array = np.array(list(sorted_node_feat_dict.values()))
    # print("ok till here")
    # print(node_feat_array)
    # print("--------------------")
    # print(graph.num_nodes())
    # print(edge_list_coo)
    graph.ndata['h'] = torch.FloatTensor(node_feat_array)
    return graph


def dataset_to_graph(dataset, dataset_name=""):
    """
    This function converts a dataset into a graph (compatible with DGL). Also calculates dataset's edgelist based on a
    distance threshold of the pairwise distances. Additionally, calculates node embeddings based on weighted deepwalk_custom.

    :param dataset_name: Used when writing the dataset
    :type dataset_name:
    :param dataset: Either the path to the dataset or the dataset as a np.array.
    :type dataset: str | pd.DataFrame |np.ndarray
    :return:
    :rtype:
    """

    if type(dataset) is str:
        dataset = pd.read_csv(dataset)
        dataset_name = dataset.split('/')[-1].split('.')[0]
    else:
        dataset = PCA(n_components=0.9).fit_transform(dataset)
        dataset_name = dataset_name

    # --- Find Edges ---
    # cosine similarity matrix, NxN with values in [0,1]
    cosine_matrix = cosine_similarity(dataset)
    edge_list = extract_edges(similarity_matrix=cosine_matrix, dataset_name=dataset_name)

    # # there are some cases where the graph created from a dataset
    # # was empty
    # # this check is applied because deepwalk_custom can't
    # # produce node embeddings for an empty graph
    if edge_list is None:
        raise Exception("Cannot convert dataset to graph. No edges found, consider adjusting the distance threshold "
                        "level.")
    else:
        extract_node_embeddings(dataset_name=dataset_name)
        # Depict graph as DGL object, from edgelist and node embeddings files.
        graph = graph_implementation(edge_list, dataset_name)
        return graph
