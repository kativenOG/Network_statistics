from sklearn.cluster import SpectralClustering,KMeans
import networkx as nx 
import numpy as np 
from sklearn import metrics

def applyKMeans(n_classes,adj_mat,gt,n_init):
    """
    Another clustering technique for comparison !!
    """
    print(f"KMeans Clustering used for comparison !!") 
    kmeans = KMeans(n_classes,n_init=n_init)
    kmeans.fit(adj_mat)
    scores = getScores(kmeans.labels_,gt,verbose = True)
    return kmeans.labels_,scores 
 
def applySpectral(n_classes,adj_mat,gt,assign_labels="kmeans",n_init=100):
    """
     Params:
     - 'precomputed' means we are using our own affinity Matrix when we call fit 
     - 'n_init' is the number of times the k-means algorithm will be run with different centroid seeds 
     - 'assign_labels':
         1. 'kmeans': is the default way the labels are assigned in the embedding space, it's sensible to initialization 
         2. 'discretize': another approach less sensitive to initialization 
         3. 'cluster_qr': directly extract clusters from eigen vectors: no tuning parameter, no runs of iterations but can outperform the other 2 in terms of quality(accuracy) of result and speed
    """

    print(f"Spectral Clustering using {assign_labels.upper()} to assign labels !!")
    if assign_labels == "kmeans": sc = SpectralClustering(n_classes, affinity='precomputed', n_init=n_init) 
    elif assign_labels == "discretize": sc = SpectralClustering(n_classes, affinity='precomputed', assign_labels=assign_labels)
    else: sc = SpectralClustering(n_classes, affinity='precomputed', assign_labels="cluster_qr") 

    sc.fit(adj_mat)
    scores = getScores(sc.labels_,gt,verbose = True)

    return sc.labels_,scores

def getScores(labels,gt,verbose = False):
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

def karateClubGenerator():
    print("KARATE CLUB GRAPH")
    G = nx.karate_club_graph()
    adj_mat = nx.to_numpy_array(G) # Return the graph adjacency matrix as a NumPy matrix.
    gt = [G.nodes[node]["club"] for node in G.nodes()]
    len_classes = len(set(gt))
    return adj_mat,len_classes,gt 

def SBMGenerator(sizes,p,sparse=False):
    print(f"{'SPARSE' if sparse else ''} STOCHASTIC BLOCK MODEL GRAPH:")
    G = nx.stochastic_block_model(sizes,p=p,sparse=sparse)
    adj_mat = nx.to_numpy_array(G) # Return the graph adjacency matrix as a NumPy matrix.
    len_classes = len(sizes)
    return adj_mat,len_classes
