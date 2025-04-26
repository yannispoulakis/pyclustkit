import networkx as nx
import numpy as np
import random

def getTransitionMatrix(network, nodes):

	nodes = [n for n in network.nodes()] # my add on
	matrix = np.zeros([len(nodes), len(nodes)])# my add on
	
	for i in range(0, len(nodes)):
		neighs = [*network.neighbors(nodes[i])]# my modification
		sums = 0
		for neigh in neighs:
			
			sums += network[nodes[i]][neigh]['weight']

		for j in range(0, len(nodes)):

			if nodes[j] in neighs: # my modification

				matrix[i, j] = network[nodes[i]][nodes[j]]['weight'] / sums
		
	return matrix

def generateSequence(startIndex, transitionMatrix, path_length, alpha):
	result = [startIndex]
	current = startIndex

	for i in range(0, path_length):
		if random.random() < alpha:
			nextIndex = startIndex
		else:
			probs = transitionMatrix[current]
			nextIndex = np.random.choice(len(probs), 1, p=probs)[0]

		result.append(nextIndex)
		current = nextIndex

	return result

def random_walk(G, num_paths, path_length, alpha):
	nodes = G.nodes()
	transitionMatrix = getTransitionMatrix(G, nodes)

	sentenceList = []

	nodes_list = [n for n in nodes]

	for i in range(0, len(nodes)):
		for j in range(0, num_paths):
			indexList = generateSequence(i, transitionMatrix, path_length, alpha)
			sentence = [int(nodes_list[tmp]) for tmp in indexList]
			sentenceList.append(sentence)

	return sentenceList