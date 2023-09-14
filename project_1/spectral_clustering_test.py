import networkx as nx 
import numpy as np 
from sklearn.cluster import SpectralClustering,KMeans
from sklearn import metrics
import random 
import os 

N_INIT = 100 

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
    len_classes = len(set(gt))
    return adj_mat,len_classes

if __name__ == "__main__":

    # Karate Club 
    adj_mat,len_classes,gt = karateClubGenerator()
    kcscores = [] 
    kcscores.append(applySpectral(len_classes,adj_mat,gt,n_init=N_INIT))
    kcscores.append(applySpectral(len_classes,adj_mat,gt,assign_labels="discretize")[1])
    kcscores.append(applySpectral(len_classes,adj_mat,gt,assign_labels="cluster_qr")[1])
    kcscores.append(applyKMeans(len_classes,adj_mat,gt,N_INIT)[1])

    # SBM: 
    # Variables: 
    n_classes = 10
    N1,N2 = 1000,10000
    sizes1,sizes2 = [N1//n_classes]*n_classes,[N2//n_classes]*n_classes
    gt1,gt2 = np.array([[val]*(N1//n_classes) for val in range(10)]), np.array([[val]*(N2//n_classes) for val in range(10)])
    gt1,gt2 = gt1.flatten(),gt2.flatten()
    # P defines if our matrix is dense or not
    dense_p = np.array([random.uniform(0,1)]*(n_classes*n_classes))
    dense_p = np.reshape(dense_p,(n_classes,n_classes))
    sparse_p1,sparse_p2 = np.array([random.uniform(0,N1)/N1] for _ in range(N1*N1)),np.array([random.uniform(0,N2)/N2] for _ in range(n_classes*n_classes))
    sparse_p1,sparse_p2 =  np.reshape(sparse_p1,(n_classes,n_classes)), np.reshape(sparse_p2,(n_classes,n_classes))

    # Dense with N1
    adj_mat,len_classes = SBMGenerator(sizes1,dense_p)
    sbmscores_dense1 = [] 
    sbmscores_dense1.append(applySpectral(len_classes,adj_mat,gt1,n_init=N_INIT))
    sbmscores_dense1.append(applySpectral(len_classes,adj_mat,gt1,assign_labels="discretize")[1])
    sbmscores_dense1.append(applySpectral(len_classes,adj_mat,gt1,assign_labels="cluster_qr")[1])
    sbmscores_dense1.append(applyKMeans(len_classes,adj_mat,gt1,N_INIT)[1]) 
    # Dense with N2
    adj_mat,len_classes = SBMGenerator(sizes2,dense_p)
    sbmscores_dense2 = [] 
    sbmscores_dense2.append(applySpectral(len_classes,adj_mat,gt2,n_init=N_INIT))
    sbmscores_dense2.append(applySpectral(len_classes,adj_mat,gt2,assign_labels="discretize")[1])
    sbmscores_dense2.append(applySpectral(len_classes,adj_mat,gt2,assign_labels="cluster_qr")[1])
    sbmscores_dense2.append(applyKMeans(len_classes,adj_mat,gt2,N_INIT)[1]) 
    # Sparse with N1
    adj_mat,len_classes = SBMGenerator(sizes1,sparse_p1)
    sbmscores_sparse1 = [] 
    sbmscores_sparse1.append(applySpectral(len_classes,adj_mat,gt1,n_init=N_INIT))
    sbmscores_sparse1.append(applySpectral(len_classes,adj_mat,gt1,assign_labels="discretize")[1])
    sbmscores_sparse1.append(applySpectral(len_classes,adj_mat,gt1,assign_labels="cluster_qr")[1])
    sbmscores_sparse1.append(applyKMeans(len_classes,adj_mat,gt1,N_INIT)[1]) 
    # Sparse with N2
    adj_mat,len_classes = SBMGenerator(sizes2,sparse_p2)
    sbmscores_sparse2 = [] 
    sbmscores_sparse2.append(applySpectral(len_classes,adj_mat,gt2,n_init=N_INIT))
    sbmscores_sparse2.append(applySpectral(len_classes,adj_mat,gt2,assign_labels="discretize")[1])
    sbmscores_sparse2.append(applySpectral(len_classes,adj_mat,gt2,assign_labels="cluster_qr")[1])
    sbmscores_sparse2.append(applyKMeans(len_classes,adj_mat,gt2,N_INIT)[1]) 

