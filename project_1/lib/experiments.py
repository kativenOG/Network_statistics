import random 
import numpy as np 
from spectral import *

def karate_club_test(n_init):
    """
    Karate Club 
    """
    adj_mat,len_classes,gt = karateClubGenerator()
    scores = [] 
    scores.append(applySpectral(len_classes,adj_mat,gt,n_init=n_init))
    scores.append(applySpectral(len_classes,adj_mat,gt,assign_labels="discretize")[1])
    scores.append(applySpectral(len_classes,adj_mat,gt,assign_labels="cluster_qr")[1])
    scores.append(applyKMeans(len_classes,adj_mat,gt,n_init)[1])
    return scores 

def generateSBMParams(n_classes,n):
    """
    Stocastic Block Model Params  
    """
    sizes,gts,sparse_ps = [],[],[]
    # Params for each and every n (number of vertices)
    for val in n:
        sizes.append([val//n_classes]*n_classes)
        gt = np.array([[val]*(val//n_classes) for val in range(n_classes)])
        gts.append(gt.flatten())
        sparse_p = np.array([random.uniform(0,val)/val for _ in range(pow(n_classes,2))])
        sparse_ps.append(np.reshape(sparse_p,(n_classes,n_classes)))
    # Calculate P for Dense matrices 
    dense_p = np.array([random.uniform(0,1)]*(n_classes*n_classes))
    dense_p = np.reshape(dense_p,(n_classes,n_classes))
    return tuple(sizes),tuple(sparse_ps),tuple(gts),dense_p

def sbm_test(sizes,P,gt,n_init):
    adj_mat,len_classes = SBMGenerator(sizes,P)
    scores = [] 
    scores.append(applySpectral(len_classes,adj_mat,gt,n_init=n_init))
    scores.append(applySpectral(len_classes,adj_mat,gt,assign_labels="discretize")[1])
    scores.append(applySpectral(len_classes,adj_mat,gt,assign_labels="cluster_qr")[1])
    scores.append(applyKMeans(len_classes,adj_mat,gt,n_init)[1]) 
    return scores 

