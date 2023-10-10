import networkx as nx 
import numpy as np 
import os,shutil,subprocess

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

# if __name__ == "__main__":
#     k_folds = jazz_generator() 
#     print(f"Number of fodls: {len(k_folds)}")
#     for fold in k_folds:
#         print(fold.shape)
    # print(f"Graph:\n{G}\nADJ:\n{adj}")
