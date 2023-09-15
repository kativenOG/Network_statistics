from networkx.algorithms.bipartite import color
from sklearn.cluster import SpectralClustering,KMeans
import matplotlib.pyplot as plt
import networkx as nx 
from sklearn import metrics
import os 

def applyKMeans(n_classes,adj_mat,gt,n_init,verbose=True):
    """
    Another clustering technique for comparison !!
    """
    if verbose: print(f"KMeans Clustering used for comparison !!") 
    kmeans = KMeans(n_classes,n_init=n_init)
    kmeans.fit(adj_mat)
    scores = getScores(kmeans.labels_,gt,verbose = verbose)
    return scores 
 
def applySpectral(n_classes,adj_mat,gt,assign_labels="kmeans",n_init=100,verbose=True):
    """
     Params:
     - 'precomputed' means we are using our own affinity Matrix when we call fit 
     - 'n_init' is the number of times the k-means algorithm will be run with different centroid seeds 
     - 'assign_labels':
         1. 'kmeans': is the default way the labels are assigned in the embedding space, it's sensible to initialization 
         2. 'discretize': another approach less sensitive to initialization 
         3. 'cluster_qr': directly extract clusters from eigen vectors: no tuning parameter, no runs of iterations but can outperform the other 2 in terms of quality(accuracy) of result and speed
    """

    if verbose: print(f"Spectral Clustering using {assign_labels.upper()} to assign labels !!")
    if assign_labels == "kmeans": sc = SpectralClustering(n_classes, affinity='precomputed', n_init=n_init) 
    elif assign_labels == "discretize": sc = SpectralClustering(n_classes, affinity='precomputed', assign_labels=assign_labels)
    else: sc = SpectralClustering(n_classes, affinity='precomputed', assign_labels="cluster_qr") 

    sc.fit(adj_mat)
    scores = getScores(sc.labels_,gt,verbose =verbose)

    return scores

def getScores(labels,gt,verbose = True):
    # Calculate some Clustering Metrics
    ari = metrics.adjusted_rand_score(gt, labels)
    ami = metrics.adjusted_mutual_info_score(gt, labels)
    mi  = metrics.mutual_info_score(gt, labels) # Standard Mutual Information 
    ri  = metrics.rand_score(gt, labels) # Standard Rand Score
    
    if verbose: 
        print("Scores:")
        print(f"\tMutual Information: {mi:.4}\n\tAdjusted Mutual Information: {ami:.4}")
        print(f"\tRand Score: {ri:.4}\n\tAdjusted Rand Score: {ari:.4}\n")
    return tuple([ri,ari]),tuple([mi,ami])

def plot_scores(score_name, x_labels, data,output_dir, file_name):
    colors = ["black","blue","green","red","lime","darkorange","tan","gray","peru"]
    fig, (ax1,ax2) = plt.subplots(1,2) #,sharex=True) 
    fig.suptitle(score_name) 
    ax1.set_title(score_name.splitlines()[1].split("and")[0])
    ax2.set_title(score_name.splitlines()[1].split("and")[1])
    # list(range(1,len(data)+1))
    ax1.bar(x=x_labels,height=list(map(lambda x: x[0],data)),color=colors[:len(data)]) 
    ax2.bar(x=x_labels,height=list(map(lambda x: x[1],data)),color=colors[:len(data)]) 
    plt.savefig(os.path.join(output_dir,file_name))
    plt.show()
    return 

def karateClubGenerator(verbose=True):
    if verbose: print("KARATE CLUB GRAPH")
    G = nx.karate_club_graph()
    adj_mat = nx.to_numpy_array(G) # Return the graph adjacency matrix as a NumPy matrix.
    gt = [G.nodes[node]["club"] for node in G.nodes()]
    gt = [0 if node=="Mr. Hi" else 1 for node in gt]
    len_classes = len(set(gt))
    return adj_mat,len_classes,gt 

def SBMGenerator(sizes,p,sparse=False,verbose=True):
    if verbose: print(f"{'SPARSE' if sparse else ''} STOCHASTIC BLOCK MODEL GRAPH:")
    G = nx.stochastic_block_model(sizes,p=p,sparse=sparse)
    adj_mat = nx.to_numpy_array(G) # Return the graph adjacency matrix as a NumPy matrix.
    len_classes = len(sizes)
    return adj_mat,len_classes

