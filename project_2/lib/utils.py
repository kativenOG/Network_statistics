import networkx as nx 
import numpy as np 
import os,shutil,subprocess
# import pandas as pd 

def jazz_generator():
    # Get rid of old repositories if present (just to be sure)
    dir1,dir2  =  os.path.join(os.getcwd(),"{}"),os.path.join(os.getcwd(),"dimacs10-netscience")
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
    
    # Generate Networkx Objects
    G = nx.from_edgelist(edge_list)
    adj_mat = nx.to_numpy_array(G)  # Return the graph adjacency matrix as a NumPy matrix.
    return G,adj_mat
    
if __name__ == "__main__":
    G,adj = jazz_generator() 
    print(f"Graph:\n{G}\nADJ:\n{adj}")
