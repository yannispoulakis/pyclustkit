import numpy as np
import torch
import os

core_dir = os.path.dirname(os.path.dirname(__file__))

def _ne_file2torch(path):
    """
    Transforms a node embeddings file to a torch tensor.
    :return:
    :rtype:
    """
    path = 'ne_' + path + '.file'
    path = os.path.join(core_dir, "temp", "embeddings", path)
    with open(path, 'r') as f:
        line = f.readline()
        line = f.readline()
        node_feat_dict = {}
        while line:
            cur_line = line.split(" ")
            node_feat_dict[int(cur_line[0])] = [float(val) for val in cur_line[1:]]
            line = f.readline()

    sorted_node_feat_dict = dict(sorted(node_feat_dict.items(), key=lambda e: e[0]))
    node_feat_array = np.array(list(sorted_node_feat_dict.values()))
    return torch.FloatTensor(node_feat_array)