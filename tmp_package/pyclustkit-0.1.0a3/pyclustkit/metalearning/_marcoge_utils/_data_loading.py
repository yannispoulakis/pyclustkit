import torch
from torch.utils.data import Dataset
from dgl.dataloading import GraphDataLoader


# create a Pytorch Dataset of graphs
class GraphAndLabelDataset(Dataset):
    def __init__(self, graph_list, labels, mapping_indices):
        """
        Initialize the GraphAndLabelDataset.

        Args:
            graph_list (list of DGLGraph): A list of DGL graphs.
            labels (list or torch.Tensor): A list or tensor of labels for the graphs.
        """
        self.graph_list = graph_list
        self.labels = torch.tensor(labels)  # Convert labels to a tensor
        self.mapping_indices = torch.tensor(mapping_indices)

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        graph = self.graph_list[idx]
        label = self.labels[idx]
        mapping_index = self.mapping_indices[idx]

        return graph, label, mapping_index


# creation of graphs-labels dataloaders
def create_dataloader(graphs, labels, mapping):

    graph_dataset = GraphAndLabelDataset(graphs, labels, mapping)

    dataloader = GraphDataLoader(graph_dataset, batch_size = 16, shuffle= False, drop_last = False)
    return dataloader
