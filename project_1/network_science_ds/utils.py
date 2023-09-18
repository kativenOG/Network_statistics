import csv,subprocess,os,shutil
import networkx as nx 
import matplotlib.pyplot as plt

def generate_ns_ds_graph():
    dir =  os.path.join(os.getcwd(),"{}")
    if os.path.isdir(dir): shutil.rmtree(dir)
    subprocess.run(["wget","-P","{}","http://konect.cc/files/download.tsv.dimacs10-netscience.tar.bz2"])
    # G2 =  nx.read_adjlist("{}/download.tsv.dimacs10-netscience.tar.bz2","rb")
    subprocess.run(["bzip2","-d","{}/download.tsv.dimacs10-netscience.tar.bz2"])
    subprocess.run(["tar","-xf","{}/download.tsv.dimacs10-netscience.tar"])
     
    with open("dimacs10-netscience/out.dimacs10-netscience","r") as f:
        lines = f.readlines()        
    with open("dimacs10-netscience/out.dimacs10-netscience","w") as f:
        for line in lines[1:]: f.write(line)

    G =  nx.read_adjlist("dimacs10-netscience/out.dimacs10-netscience")
    subprocess.run(["rm","-rf","{}","dimacs10-netscience"])
    return G 

graph = generate_ns_ds_graph()
print(graph)

