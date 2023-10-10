import networkx as nx 
import numpy as np 
import os,shutil,subprocess,random

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
    print(f"Number of fodls: {len(k_folds)}")
    for fold in k_folds:
        print(fold.shape)
    # print(f"Graph:\n{G}\nADJ:\n{adj}")
