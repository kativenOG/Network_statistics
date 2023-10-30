import networkx as nx
from networkx.algorithms.bipartite import degrees 
import numpy as np 
from collections import Counter

# Spectral sutff
import warnings 
from scipy.sparse.linalg import eigs
from .local_similarity import sort_scores
from numpy import linalg as LA 
from scipy.spatial.distance import pdist

# Plot Stuff 
import matplotlib.pyplot as plt 
import matplotlib as mpl

def biggest_clique_removal(k_folds: list, n_cliques: int =1, reverse: bool=True)-> tuple[nx.Graph,list]:
    G = nx.from_edgelist(np.vstack(k_folds))

    cliques = list(map(lambda x: set(x),nx.find_cliques(G)))
    cliques.sort(key=lambda x: len(x),reverse=reverse)

    biggest_cliques = cliques[:n_cliques]
    union_of_cliques = set()
    probe_set = list(union_of_cliques.union(*biggest_cliques))

    clique_edge_list = list(G.subgraph(probe_set).edges())
    clique_edge_list = [ edge[::-1] for edge in clique_edge_list] + clique_edge_list
    
    # Only removing clique Edges from OG graph 
    full_set = set(G.edges())
    correct_set = list(full_set.difference(set(clique_edge_list)))

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



# Poster Function Plots 

def draw_clique_size_frequency(k_folds: list)->Counter:
    """
    Draw a bar graph the most frequent size of the graph's cliques.
    """
    # Get Clique Data 
    G = nx.from_edgelist(np.vstack(k_folds))
    cliques_sizes = list(map(lambda x: len(set(x)),nx.find_cliques(G)))
    counter = Counter(cliques_sizes) 
    
    # Plot Stuff:
    plt.figure(facecolor="#eeeeee")
    plt.bar(counter.keys(),counter.values())

    plt.title("Cliques Size Frequency", fontweight='bold')
    plt.xlabel('Clique Size', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')

    plt.show()
    return counter

def draw_degree_distribution(k_folds: list)->Counter:
    """
    Draw a bar graph in wich each column represent a degree value and the y's are the node's degree frequency  
    """
    # Get Degree Data 
    G = nx.from_edgelist(np.vstack(k_folds))
    node_degrees = dict(G.degree()).values()
    counter = Counter(node_degrees) 
    
    # Plot Stuff:
    plt.figure(facecolor="#eeeeee")
    plt.bar(counter.keys(),counter.values())

    plt.title("Degree Distribution", fontweight='bold')
    plt.xlabel('Node degree', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')

    plt.show()
    return counter

def generate_color_map(values:list)-> list:
    color_lookup =  list(set(values)) # Removed sorted

    low, high  = color_lookup[0],color_lookup[-1]
    norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)
    return [mapper.to_rgba(val) for val in values]


def draw_cliques(k_folds: list,reverse:bool=False,n_selection:int=10, center=False)->None:

    # Full Graph 
    G = nx.from_edgelist(np.vstack(k_folds))

    # Background Colorfor the plot 
    plt.axis('equal') 
    ax1 = plt.axes()
    ax1.set_facecolor("black")     

    cliques = list(map(lambda x: set(x),nx.find_cliques(G)))
    cliques.sort(key=lambda x: len(x),reverse=reverse)
    selected = cliques[:n_selection] if (not center) else cliques[(len(cliques)//2)-n_selection:(len(cliques)//2)+n_selection] 
    union_of_cliques = set()
    all_clique_nodes = list(union_of_cliques.union(*selected)) 
    subgraph = G.subgraph(all_clique_nodes)
    cliques_colors = generate_color_map(list(range(len(selected))))

    # Get Node Colors Array
    node_colors = {}
    for i,color in enumerate(cliques_colors):
        for node in selected[i]: node_colors[node] = color
    node_colors = list(node_colors.values())

    # Get Edge Colors Array
    std_color = tuple([1.,1.,1.,1.0])
    edge_colors, check= [],False

    for edge in subgraph.edges():
        for i,clique in enumerate(selected):
            if (edge[0] in clique) and (edge[1] in clique): 
                edge_colors.append(cliques_colors[i])
                check=True
                break # Skip the other cliques 

        # Else if this doesnt happen 
        if not check: edge_colors.append(std_color)
        check=False # Always put check back to False 

    # Finally draw the artificial network  
    nx.draw_networkx(subgraph,
                     with_labels= False,
                     edge_color=  edge_colors, 
                     node_color= tuple([1.,1.,1.,1.]),
                     width=2.0,
                     pos=nx.spring_layout(G,k=2)) 

    plt.show()
    
    return 
    

# For testing 
# from utils import jazz_generator

# if __name__ == "__main__":
#     k_folds = jazz_generator() 
#     # _ = draw_clique_size_frequency(k_folds)
#     # _ = draw_degree_distribution(k_folds)

#     # Graph plot 
#     draw_cliques(k_folds) 
#      

