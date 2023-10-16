from utils import jazz_generator

import networkx as nx 
import numpy as np 


def biggest_clique_removal(k_folds:list, n_cliques:int =1)-> nx.Graph:
    G = nx.from_edgelist(np.vstack(k_folds) )

    cliques = list(map(lambda x: set(x),nx.find_cliques(G)))
    cliques.sort(key=lambda x: len(x),reverse=True)

    biggest_cliques = cliques[:n_cliques]
    full_set = set(G.nodes())
    correct_set = full_set.difference(*biggest_cliques) 

    return G.subgraph(correct_set) 

    
    
if __name__ == "__main__":
    k_folds = jazz_generator() 
    G = biggest_clique_removal(k_folds)
