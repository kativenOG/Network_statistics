import networkx as nx 
import numpy as np 
from sklearn.cluster import SpectralClustering,KMeans
from sklearn import metrics
import os 

N_INIT = 100 

def applyKMeans(n_classes,adj_mat,n_init):
    """
    Another clustering technique for comparison !!
    """
    kmeans = KMeans(n_classes,n_init=n_init)
    kmeans.fit(adj_mat)
    print(f"KMeans Clustering used for comparison !!") 
    return kmeans.labels_ 
 
def applySpectral(n_classes,adj_mat,assign_labels="kmeans",n_init=100):
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

    return sc.labels_ 
    

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


def karateClubGeneration():
    print("KARATE CLUB GRAPH")
    G = nx.karate_club_graph()
    adj_mat = nx.to_numpy_array(G) # Return the graph adjacency matrix as a NumPy matrix.
    gt = [G.nodes[node]["club"] for node in G.nodes()]
    len_classes = len(set(gt))
    return adj_mat,len_classes,gt 

def sparseERGraphGeneration(n=1000,p=0.03):
    print("SPARSE ERDOS RENYI GRAPH:")
    G = nx.fast_gnp_random_graph(n,p)
    adj_mat = nx.to_numpy_array(G) # Return the graph adjacency matrix as a NumPy matrix.
    gt = [G.nodes[node]["club"] for node in G.nodes()]
    len_classes = len(set(gt))
    return adj_mat,len_classes,gt 

def ERGraphGeneration(n=1000,p=0.2):
    print("ERDOS RENYI GRAPH:")
    G = nx.erdos_renyi_graph(n,p)
    adj_mat = nx.to_numpy_array(G) # Return the graph adjacency matrix as a NumPy matrix.
    gt = [G.nodes[node]["club"] for node in G.nodes()]
    len_classes = len(set(gt))
    return adj_mat,len_classes,gt 

if __name__ == "__main__":

    # Karate Club 
    adj_mat,len_classes,gt = karateClubGeneration()
    labels = applySpectral(len_classes,adj_mat,n_init=N_INIT)
    ri, mi = getScores(labels=labels,gt=gt,verbose=True)
    labels = applySpectral(len_classes,adj_mat,assign_labels="discretize")
    ri, mi = getScores(labels=labels,gt=gt,verbose=True)
    labels = applySpectral(len_classes,adj_mat,assign_labels="cluster_qr")
    ri, mi = getScores(labels=labels,gt=gt,verbose=True)
    labels = applyKMeans(len_classes,adj_mat,n_init=N_INIT)
    ri, mi = getScores(labels=labels,gt=gt,verbose=True) 

    

