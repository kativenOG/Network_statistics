from os.path import isdir
import networkx as nx 
import numpy as np 
import os,shutil,subprocess,random
import matplotlib.pyplot as plt
import matplotlib as mpl
from networkx_viewer import Viewer

def jazz_generator(k=10):
    """
    Download the dataset and return it as a list of k folds 
    """
    # Get rid of old repositories if present (just to be sure)
    dir1,dir2  =  os.path.join(os.getcwd(),"{}"),os.path.join(os.getcwd(),"jazz.net")
    if os.path.isdir(dir1): shutil.rmtree(dir1)
    if os.path.isdir(dir2): shutil.rmtree(dir2)

    # Download the dataset and unzip it 
    subprocess.run(["wget","--no-check-certificate","-P","{}","https://deim.urv.cat/~alexandre.arenas/data/xarxes/jazz.zip"])
    subprocess.run(["unzip","{}/jazz.zip"])
     
    # Save all the lines in the Graph
    with open("jazz.net","r") as f: lines = f.readlines()        
    # Get rid of the Data 
    subprocess.run(["rm","-rf","{}","jazz.net"]) 
    
    # Transformation to edge list
    data= [list(map(lambda x: int(x),line.split())) for line in lines[3:]]
    dim1, dim2=len(data),len(data[0])
    array = np.array(data)
    edge_list = np.reshape(array,(dim1,dim2))[:,:2]
     
    # Shuffle the rows :) 
    np.random.shuffle(edge_list)
    
    # Split into k folds
    SHAPE = edge_list.shape
    fold_size = SHAPE[0]// k
    k_folds = []
    for _ in range(10):
        k_folds.append(edge_list[:fold_size,:])
        edge_list = edge_list[fold_size:,:]

    return k_folds    

def use_newtorkx_viewer(k_folds):
    # Generate the Graph from k_folds 
    edge_list = np.vstack(k_folds)
    G = nx.from_edgelist(edge_list) 
    # View it 
    app = Viewer(G)
    app.mainloop()

def get_max_connected_component(k_folds)-> nx.Graph:
     """
     There is only one connected componet so the function is useless !
     """
     full_edge_list = np.vstack(k_folds) 
     full_G = nx.from_edgelist(full_edge_list)
     print(f"The number of connected components is: {len(list(nx.connected_components(full_G)))}\n")
     max_cc = max(nx.connected_components(full_G))
     G = nx.subgraph(full_G,max_cc)
     return G 

def plot_full_graph(k_folds,with_labels=False,edge_color = "white",using_ipynb=False,save_plt=True):#,pos=nx.spring_layout):
    # Generate the Graph from k_folds and return only biggest cc (for this dataset is only 1 cc) 
    G = get_max_connected_component(k_folds)

    # Image figure 
    fig = None
    fig = plt.figure(figsize=(50,50),facecolor="black") # In inches, !!! Doesn't Work !!! 

    # Change plt background color and size
    plt.axis('equal') 
    ax1 = plt.axes()
    ax1.set_facecolor("black")     

    
    # Set node sizes based on node degree
    node_sizes = list(map(lambda x: x[1]+25,G.degree()))

    # Assign node color based on the degree values
    color_lookup =  sorted(set(node_sizes))#{k:v for v, k in enumerate(sorted(set(gt)))}
    low, high  = color_lookup[0],color_lookup[-1]
    norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)
    node_color = [mapper.to_rgba(val) for val in node_sizes]
    
    # Plot graph 
    nx.draw_networkx(G,node_size=node_sizes,
                     with_labels=with_labels,
                     edge_color=edge_color,
                     node_color=node_color,
                     pos=nx.spring_layout(G,k=2)) #=pos(G))
    if using_ipynb: plt.show
    else: plt.show()

    # Save plt
    if save_plt:
        if not os.path.isdir("figures"): os.mkdir("figures")
        if fig!=None:
            fig.savefig(os.path.join("figures","graph.png")) 
            plt.savefig(os.path.join("figures","graph.svg")) 
        else:
            plt.savefig(os.path.join("figures","graph.png"), dpi=1000)
            plt.savefig(os.path.join("figures","graph.pdf")) 

def train_probe_split(k_folds,probe_index):
    train_set,probe_set  = np.zeros((1,k_folds[0].shape[1])),0
    for index,fold in enumerate(k_folds): 
        if index==probe_index: probe_set = fold
        else:  
            train_set = np.vstack(tup=(train_set,fold))
    return train_set[1:,:],probe_set

def from_el_to_nx(edge_list):
    G = nx.from_edgelist(edge_list)
    adj_mat = nx.to_numpy_array(G)  # Return the graph adjacency matrix as a NumPy matrix.
    return G,adj_mat

def accuracy_metric(scores,probe_set):
    L = probe_set.shape[0] 
    top_scores = np.array(scores)[:L,:2]
    l = sum([1 for edge_pair in probe_set if (edge_pair == top_scores).all(axis=1).any()])
    return l/L 

def auc_metric(scores,train_set,probe_set,n=100,seed=1234):
    len_probe = probe_set.shape[0]
    if n>len_probe: n=len_probe 
    random.seed(a=1234, version=2)
    
    # Getting the whole graph
    complete_edge_list = np.vstack((train_set,probe_set))
    G = nx.from_edgelist(complete_edge_list)
    non_edges = list(nx.non_edges(G))
    len_non_edges= len(non_edges)

    n_1,n_2 = 0,0  
    for _ in range(n):
        # Get two different random edges
        while True: 
            random_probe = probe_set[random.randrange(0,len_probe)]
            random_non_edge = non_edges[random.randrange(0,len_non_edges)]
            if (random_probe!=random_non_edge).all() and (np.flip(random_probe)!=random_non_edge).all(): break
        
        # Get the scores for the two random edges
        probe_score,non_edge_score = 0,0
        for u,v,score in scores:

            if (u in random_probe) and (v in random_probe): probe_score = score 
            elif (u in random_non_edge) and (v in random_non_edge): non_edge_score = score 
            
            # Exit condition when both of them are found 
            if (probe_score != 0) and (non_edge_score!=0): break
        
        # Increase counters
        if probe_score > non_edge_score: n_1+=1  
        elif probe_score == non_edge_score: n_2+=1   

    return (n_1+(0.5*n_2))/n


##### For Testing ###########
if __name__ == "__main__":
    k_folds = jazz_generator() 
    plot_full_graph(k_folds) 
    use_newtorkx_viewer(k_folds) 
