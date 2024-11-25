#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
from io import open
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import logging

from  ..deepwalk_custom import graph
from  ..deepwalk_custom import walks as serialized_walks
from ..deepwalk_custom.skipgram import Skipgram

from gensim.models import Word2Vec


from six import text_type as unicode
from six import iteritems
#from six.moves import range

import psutil
from multiprocessing import cpu_count

import networkx as nx
from deepwalk import weighted_random_walk

p = psutil.Process(os.getpid())
p.cpu_affinity(list(range(cpu_count())))

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def debug(type_, value, tb):
  if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    sys.__excepthook__(type_, value, tb)
  else:
    import traceback
    import pdb
    traceback.print_exception(type_, value, tb)
    print(u"\n")
    pdb.pm()


def process(args):

  if args.format == "adjlist":
    G = graph.load_adjacencylist(args.input, undirected=args.undirected)
  elif args.format == "edgelist":
    G = graph.load_edgelist(args.input, undirected=args.undirected)
  elif args.format == "mat":
    G = graph.load_matfile(args.input, variable_name=args.matfile_variable_name, undirected=args.undirected)
  elif args.format == 'weighted_edgelist':
    G = nx.read_weighted_edgelist(args.input, nodetype = int) #my modification
  else:
    raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', 'mat'" % args.format)

  print("Number of nodes: {}".format(len(G.nodes())))

  num_walks = len(G.nodes()) * args.number_walks

  print("Number of walks: {}".format(num_walks))

  data_size = num_walks * args.walk_length

  print("Data size (walks*length): {}".format(data_size))

  if data_size < args.max_memory_data_size:
    print("Walking...")
    if args.format == 'weighted_edgelist':
      #only changaed this part -- shun
      walks = weighted_random_walk.random_walk(G, num_paths=args.number_walks,path_length=args.walk_length, alpha=0)
    else:
      walks = graph.build_deepwalk_corpus(G, num_paths=args.number_walks,path_length=args.walk_length, alpha=0, rand=random.Random(args.seed))
 
    print("Training...")
    #modify from size to vector_size
    # argument name 
    model = Word2Vec(walks, vector_size=args.representation_size, window=args.window_size, min_count=0, workers=args.workers)

  model.wv.save_word2vec_format(args.output) #modification


def main():
  parser = ArgumentParser("deepwalk_custom",
                          formatter_class=ArgumentDefaultsHelpFormatter,
                          conflict_handler='resolve')

  parser.add_argument("--debug", dest="debug", action='store_true', default=False,
                      help="drop a debugger if an exception is raised.")

  parser.add_argument('--format', default='adjlist',
                      help='File format of input file')

  parser.add_argument('--input', nargs='?', required=True,
                      help='Input graph file')

  parser.add_argument("-l", "--log", dest="log", default="INFO",
                      help="log verbosity level")

  parser.add_argument('--matfile-variable-name', default='network',
                      help='variable name of adjacency matrix inside a .mat file.')

  parser.add_argument('--max-memory-data-size', default=1000000000, type=int,
                      help='Size to start dumping walks to disk, instead of keeping them in memory.')

  parser.add_argument('--number-walks', default=10, type=int,
                      help='Number of random walks to start at each node')

  parser.add_argument('--output', required=True,
                      help='Output representation file')

  parser.add_argument('--representation-size', default=64, type=int,
                      help='Number of latent dimensions to learn for each node.')

  parser.add_argument('--seed', default=0, type=int,
                      help='Seed for random walk generator.')

  parser.add_argument('--undirected', default=True, type=bool,
                      help='Treat graph as undirected.')

  parser.add_argument('--vertex-freq-degree', default=False, action='store_true',
                      help='Use vertex degree to estimate the frequency of nodes '
                           'in the random walks. This option is faster than '
                           'calculating the vocabulary.')

  parser.add_argument('--walk-length', default=40, type=int,
                      help='Length of the random walk started at each node')

  parser.add_argument('--window-size', default=5, type=int,
                      help='Window size of skipgram model.')

  parser.add_argument('--workers', default=1, type=int,
                      help='Number of parallel processes.')


  args = parser.parse_args()
  numeric_level = getattr(logging, args.log.upper(), None)
  logging.basicConfig(format=LOGFORMAT)
  logger.setLevel(numeric_level)

  if args.debug:
   sys.excepthook = debug

  process(args)

if __name__ == "__main__":
  sys.exit(main())