from  ..deepwalk_custom import graph
import networkx as nx
from ..deepwalk_custom import weighted_random_walk
from gensim.models import Word2Vec
import random


import random
import networkx as nx
from gensim.models import Word2Vec
from ..deepwalk_custom import graph
from ..deepwalk_custom.skipgram import Skipgram
from ..deepwalk_custom.weighted_random_walk import random_walk


def process_graph(input_file, output_file, format="adjlist", walk_length=40, number_walks=10,
                  representation_size=64, window_size=5, workers=1, undirected=True,
                  seed=0, matfile_variable_name="network", max_memory_data_size=1e9):
    """
    Process a graph file and train DeepWalk embeddings.

    Parameters:
    - input_file (str): Path to the input graph file.
    - output_file (str): Path to save the output embeddings.
    - format (str): Format of the graph ('adjlist', 'edgelist', 'mat', 'weighted_edgelist').
    - walk_length (int): Length of random walks.
    - number_walks (int): Number of random walks per node.
    - representation_size (int): Dimension of embeddings.
    - window_size (int): Context window size for Word2Vec.
    - workers (int): Number of worker threads.
    - undirected (bool): Whether to treat the graph as undirected.
    - seed (int): Seed for random walk generator.
    - matfile_variable_name (str): Variable name for MAT format.
    - max_memory_data_size (int): Threshold for keeping walks in memory.

    Returns:
    None
    """
    # Load the graph
    if format == "adjlist":
        G = graph.load_adjacencylist(input_file, undirected=undirected)
    elif format == "edgelist":
        G = graph.load_edgelist(input_file, undirected=undirected)
    elif format == "mat":
        G = graph.load_matfile(input_file, variable_name=matfile_variable_name, undirected=undirected)
    elif format == "weighted_edgelist":
        G = nx.read_weighted_edgelist(input_file, nodetype=int)
    else:
        raise ValueError(f"Unknown file format: {format}. Valid formats: 'adjlist', 'edgelist', 'mat', 'weighted_edgelist'")

    print(f"Number of nodes: {len(G.nodes())}")

    # Generate random walks
    if format == "weighted_edgelist":
        walks = random_walk(G, num_paths=number_walks, path_length=walk_length, alpha=0)
    else:
        walks = graph.build_deepwalk_corpus(G, num_paths=number_walks, path_length=walk_length, alpha=0, rand=random.Random(seed))

    print("Training Word2Vec model...")
    model = Word2Vec(walks, vector_size=representation_size, window=window_size, min_count=0, workers=workers)

    print(f"Saving embeddings to {output_file}...")
    model.wv.save_word2vec_format(output_file)
    print("Done.")
