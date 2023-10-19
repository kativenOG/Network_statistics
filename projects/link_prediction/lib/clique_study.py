import networkx as nx 
import numpy as np 
from icecream import ic

# Spectral sutff
import warnings 
from scipy.sparse.linalg import eigen, eigs
from .local_similarity import sort_scores
from numpy import linalg as LA 
from scipy.spatial.distance import pdist


def biggest_clique_removal(k_folds:list, n_cliques:int =1)-> tuple[nx.Graph,list]:
    G = nx.from_edgelist(np.vstack(k_folds))

    cliques = list(map(lambda x: set(x),nx.find_cliques(G)))
    cliques.sort(key=lambda x: len(x),reverse=True)

    biggest_cliques = cliques[:n_cliques]
    union_of_cliques = set()
    probe_set = list(union_of_cliques.union(*biggest_cliques))

    clique_edge_list = list(G.subgraph(probe_set).edges())

    # direct_map, inverse_map = {},{}
    # visited = set()
    # for val in probe_set:  
    #     visited.add(val)
    # i = 0
    # for val in sorted(list(visited)):
    #     i+=1
    #     direct_map[i] = val 
    #     inverse_map[val ] = i
    # clique_edge_list = list(map(lambda x: list([direct_map[x[0]], direct_map[x[1]]]),clique_edge_list))
    
    ic(len(clique_edge_list))
    clique_edge_list = [ edge[::-1] for edge in clique_edge_list] + clique_edge_list

    # Removing both clique Nodes and Edges from OG graph 
    # full_set = set(G.nodes())
    # correct_set = full_set.difference(*biggest_cliques) 

    # Only removing clique Edges from OG graph 
    full_set = set(G.edges())
    correct_set = list(full_set.difference(set(clique_edge_list)))
    ic(len(full_set))
    ic(len(correct_set))
    ic(len(full_set)-len(correct_set))

    return nx.from_edgelist(correct_set),clique_edge_list

    
def spectral_scores(G: nx.Graph,n_class:int =20,eigen_gap:int=-1)-> np.ndarray:
    # So warnings wont bother us :^) 
    warnings.filterwarnings("ignore") 

    # Get Standard Laplacian 
    laplacian = nx.laplacian_matrix(G)
    laplacian = laplacian.asfptype()

    # Solve the Eigen problem 
    vals_disjoint, vecs_disjoint = eigs(laplacian,n_class,which='SR')
    vals_disjoint = np.sort(vals_disjoint,).reshape(len(vals_disjoint),1)
    vals_disjoint, vecs_disjoint = np.real(vals_disjoint), np.real(vecs_disjoint) # Remove the immaginary part   

    # Eigen Gap Prompt:
    if eigen_gap == -1:
        print("Apply eigen Gap: ".upper())
        print("Eigen Values:")
        print("\n".join(list(map(lambda x : f"-{x[0]}: {x[1]} ",enumerate(vals_disjoint.flatten())))))
        slice_line = ""
        while not slice_line.isdigit(): slice_line = input("Insert the line where the eigen values need to be sliced: ")
        slice_line = int(slice_line)
        vecs_disjoint = vecs_disjoint[:,:slice_line]
    else:
        vecs_disjoint = vecs_disjoint[:,:eigen_gap] 

    # Return the eigen vectors that will work as a score for each ndoe :^)  
    return vecs_disjoint 

def spectral_similarity(G: nx.Graph,n_class:int =20,reverse:bool = False,eigen_gap:int=-1)-> list[tuple[int,int,float]]:
    # Maps for the two node numberings spectral_scores/without_clique
    mapping = { node:i for i,node in  enumerate(G.nodes())}

    # Get the spectral scores and all the non-edges
    ss = spectral_scores(G,n_class,eigen_gap)

    # Normalize the eigen vectors values 
    # for i in range(len(ss)): ss[i,:]= ss[i,:] / np.sum(ss[i,:])

    # Get all the non-edges
    non_edges = nx.non_edges(G)
    
    # Average Distance used to normalize the value 
    avg_dist = np.mean(pdist(ss).flatten())

    def formula(x: np.ndarray,y: np.ndarray)->np.float64:
        # cn = len(list(nx.common_neighbors(G,x,y)))
        norm = LA.norm(x-y)
        similarity = 1/norm
        # similarity = cn/norm
        return similarity


    def old_formula(x: np.ndarray,y: np.ndarray)->float:
        cn = len(list(nx.common_neighbors(G,x,y)))
        norm_x,norm_y = LA.norm(x),LA.norm(y)
        cosine_similarity = np.dot(ss[mapping[x],:],ss[mapping[y],:])/(norm_x*norm_y)
        return cn*cosine_similarity
     
    # Calculate the Scores 
    spectral_score = (tuple([u,v,formula(u,v)*avg_dist]) for u,v in non_edges)
    return sort_scores(spectral_score,reverse=reverse) 

# For testing functions 
# from utils import jazz_generator

# if __name__ == "__main__":
#     k_folds = jazz_generator() 
#     G,probe_set = biggest_clique_removal(k_folds)
#     spectral_similarities = spectral_similarity(G,20)
#     for x in spectral_similarities:
#         print(f"{x[0]}-{x[1]}: {x[2]}")
