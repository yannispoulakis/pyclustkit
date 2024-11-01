# priority: Marco extractor support for pd.dataframe/np.ndarray, currently supports only string
from operator import itemgetter

import numpy as np
from sklearn.model_selection import RepeatedKFold

import pandas as pd
import torch
import os

from ..metalearning._marcoge_utils._data_loading import create_dataloader
from ..metalearning._marcoge_utils.model_execution import run_one_training
from ..metalearning._marcoge_utils._graph_generation import extract_edges, dataset_to_graph
from ..metalearning._marcoge_utils.utils import _ne_file2torch
from ..metalearning._marcoge_utils.graph_classification_model import GraphConvClassifier

core_dir = os.path.dirname(os.path.dirname(__file__))

# create a model and metafeatures for each one of the cvis
def metafeature_extractor_train(graphs, labels, mapping_list,
                        epochs): # , metafeatures_path, model_parameters_path): # path_graph_dataset,

    # parent_dir = os.path.abspath(os.path.join(path_graph_dataset, os.pardir))
    # path_to_save_embeddings = os.path.join(parent_dir, metafeatures_path)
    # path_to_save_model_params = os.path.join(parent_dir, model_parameters_path)
    # for paths_to_create in [path_to_save_embeddings, path_to_save_model_params]:
      #if not os.path.exists(paths_to_create):
          #os.mkdir(paths_to_create)

    # cvis = ['AVG', 'BP', 'CH', 'DB', 'DU', 'HKK', 'HL', 'MC', 'Scat', 'SIL', 'Xie']
    #for cvi in cvis:

      #print(f'Get Embeddings for CVI {cvi}')

      #dataset_path = os.path.join(path_graph_dataset, 'graph_data_'+cvi+".dgl")
      #graphs, labels_dict = dgl.load_graphs(dataset_path)
      # labels, mapping_list = labels_dict['glabel'], labels_dict['mapping_index']

    embeddings_dict = {}
    #RepeatedKFold
    rkf = RepeatedKFold(n_splits=5, n_repeats=1)
    for train_index, test_index in rkf.split(graphs):
        train_graphs, train_labels, train_mapping = (itemgetter(*train_index)(graphs), itemgetter(*train_index)(labels),
                                                     itemgetter(*train_index)(mapping_list))
        test_graphs, test_labels, test_mapping = (itemgetter(*test_index)(graphs), itemgetter(*test_index)(labels),
                                                  itemgetter(*test_index)(mapping_list))
        training_dataloader = create_dataloader(train_graphs, train_labels, train_mapping)
        test_dataloader = create_dataloader(test_graphs, test_labels, test_mapping)

        model_params, embeddings_dict = run_one_training(training_dataloader, test_dataloader, epochs, embeddings_dict)

    cvi_embeddings = pd.DataFrame.from_dict(embeddings_dict, orient='index')
    cvi_embeddings.rename(columns={300:'label'}, inplace=True)
    # cvi_embeddings_csv = os.path.join(path_to_save_embeddings, "metafeatures_"+cvi+".csv")
    # cvi_embeddings.to_csv(cvi_embeddings_csv)

    # save model's parameters
    # model_params_path = os.path.join(path_to_save_model_params, "model_parameters_"+cvi+".pth")
    # torch.save(model_params, model_params_path)
    return cvi_embeddings


def marcoge_mf(dataset, train=False):
    if type(dataset) is str:
        df = pd.read_csv(dataset)
        dataset_name = dataset.split("\\")[-1]
    else:
        assert type(dataset) in [pd.DataFrame,np.ndarray], "Dataset type not supported"
        df = dataset
        dataset_name = r'_temp'

    graph_df = dataset_to_graph(df, dataset_name=dataset_name)

    # model = torch.load(os.path.join(core_dir, "metalearning", "temp", "gnn_model.pth"))
    model = GraphConvClassifier(in_dim = 64,
                            hidden_dim = 300,
                            n_classes = 13)
    model.load_state_dict(torch.load(os.path.join(core_dir, "metalearning", "temp", "gnn_model_state.pth")))
    model.eval()
    graph_node_embeddings = _ne_file2torch(dataset_name)
    prediction = model(graph_df, torch.rand((df.shape[0], 64)))
    mf_names = [f"marcoge_mf_{i}" for i in range(1,301)]
    return dict(zip(mf_names, prediction[1].tolist()[0]))
