import random 
import numpy as np 
from .spectral import *

def karate_club_test(n_init,verbose=True):
    """
    Karate Club main function 
    """
    adj_mat,len_classes,gt = karateClubGenerator(verbose)
    scores = [] 
    scores.append(applySpectral(len_classes,adj_mat,gt,n_init=n_init,verbose=verbose))
    scores.append(applySpectral(len_classes,adj_mat,gt,assign_labels="discretize",verbose=verbose))
    scores.append(applySpectral(len_classes,adj_mat,gt,assign_labels="cluster_qr",verbose=verbose))
    scores.append(applyKMeans(len_classes,adj_mat,gt,n_init,verbose=verbose))
    return scores 

def generateSBMParams(n_classes,n):
    """
    Stocastic Block Model Params  
    """
    sizes,gts,sparse_ps = [],[],[]
    # Generatin Params for each n (number of vertices) provided by command line 
    for val in n:
        nodes_per_class = val//n_classes
        sizes.append([nodes_per_class]*n_classes)

        # Generating ground truths 
        gt_stack = np.empty(shape=(nodes_per_class),dtype=np.int8)
        for c in range(n_classes): gt_stack = np.vstack((gt_stack,np.full(nodes_per_class,c)),dtype=np.int8)
        gts.append(gt_stack[1:].flatten())

        # Generating sparse P matrix 
        sparse_p = np.random.random_integers(0,val,size=(n_classes,n_classes))
        sym_sparse_p = (sparse_p + sparse_p.T)/2
        sym_sparse_p = sym_sparse_p/val
        sparse_ps.append(sym_sparse_p)

    # Calculating P for Dense matrices 
    dense_p = np.array([random.uniform(0,1)]*pow(n_classes,2))
    dense_p = np.reshape(dense_p,(n_classes,n_classes))
    return tuple(sizes),tuple(sparse_ps),tuple(gts),dense_p

def sbm_test(sizes,P,gt,n_init,verbose):
    """
    Stocastic Block Model main function 
    """
    adj_mat,len_classes = SBMGenerator(sizes,P,verbose)
    scores = [] 
    scores.append(applySpectral(len_classes,adj_mat,gt,n_init=n_init,verbose=verbose))
    scores.append(applySpectral(len_classes,adj_mat,gt,assign_labels="discretize",verbose=verbose))
    scores.append(applySpectral(len_classes,adj_mat,gt,assign_labels="cluster_qr",verbose=verbose))
    scores.append(applyKMeans(len_classes,adj_mat,gt,n_init,verbose=verbose)) 
    return scores 


