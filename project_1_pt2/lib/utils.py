from collections import Counter 
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx 
import os,shutil,subprocess

def ns_generator(pruning=False,pruning_factor = 3):
    # Get rid of old repositories if present (just to be sure)
    dir1,dir2  =  os.path.join(os.getcwd(),"{}"),os.path.join(os.getcwd(),"dimacs10-netscience")
    if os.path.isdir(dir1): shutil.rmtree(dir1)
    if os.path.isdir(dir2): shutil.rmtree(dir2)

    # Download the dataset and unzip it 
    subprocess.run(["wget","-P","{}","http://konect.cc/files/download.tsv.dimacs10-netscience.tar.bz2"])
    subprocess.run(["bzip2","-d","{}/download.tsv.dimacs10-netscience.tar.bz2"])
    subprocess.run(["tar","-xf","{}/download.tsv.dimacs10-netscience.tar"])
     
    # Remove the first line from the Adjecency file
    with open("dimacs10-netscience/out.dimacs10-netscience","r") as f:
        lines = f.readlines()        
    with open("dimacs10-netscience/out.dimacs10-netscience","w") as f:
        for line in lines[1:]: f.write(line)

    # Create a graph and get rid of the dataset files
    G =  nx.read_adjlist("dimacs10-netscience/out.dimacs10-netscience")
    subprocess.run(["rm","-rf","{}","dimacs10-netscience"])
    if pruning: prune_graph(G,pruning_factor)
    adj_mat = nx.to_numpy_array(G) # Return the graph adjacency matrix as a NumPy matrix.

    return G,adj_mat

def prune_graph(G,pruning_factor=3):
    sorted_cc = sorted([component for component in nx.connected_components(G) if len(component)<pruning_factor] , key=len, reverse=True)
    for component in sorted_cc:
        # If lenght of component less than pruning factor remove all this nodes from G 
        for node in component: 
            try: 
                if G.nodes[node]: G.remove_node(node)
            except: 
                e_message = """
                            The node has already been removed and doesnt exist anymore!
                            ( shouldn't happen because we are removing connected components, 
                             so if there is a bigger subgraph these 2 should be remvoed together )
                            """
                print(e_message)

def ccn_wrapper(G):
    return nx.number_connected_components(G)

def draw_network(G,file_name,gt=None):
    if gt != None:
        color_lookup =  sorted(set(gt))#{k:v for v, k in enumerate(sorted(set(gt)))}
        low, high  = color_lookup[0],color_lookup[-1]
        norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
        mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)
        nx.draw_networkx(G,node_color= [mapper.to_rgba(val) for val in gt])
    else: nx.draw_networkx(G)

    plt.show()
    plt.savefig(file_name)


def save_counter_log(results,verbose,dir_path,log=None):
    """
    Save the results (ground truths) in a readable way using counter 
    """
    file_string,file_name = "", os.path.join(dir_path,"log.txt")
    for clustering_method,result in results.items():
        counter = Counter(result) 
        file_string+= f"{clustering_method.upper()}: {counter}\n"        
    if log: open(file_name,"w+").write(file_string)
    return 

