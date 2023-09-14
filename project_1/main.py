from lib.all import *

import numpy as np 
import random 
import argparse

parser = argparse.ArgumentParser(prog='Testing Spectral Clustering Metrics')

   

if __name__ == "__main__":

    args = args_getter()
    karate_flag,sbm_flag= False, False  
    if args.mode == "all": karate_flag,sbm_flag = True,True
    elif args.mode == "sbm": sbm_flag = True
    elif args.mode == "kc": karate_flag = True
    
    k_scores, sbm_scores = None, []
    if karate_flag: 
        k_scores = karate_club_test(args.n_init) 
    
    if sbm_flag: 
        sizes,sparse_ps,gts,dense_p =generateSBMParams(args.n_classes,args.n_vals)
        sbm_score = {} 
        for  s,sparse_p,gt in zip(sizes,sparse_ps,gts):
            sbm_score["sparse"] = sbm_test(s,sparse_p,gt,args.n_init)
            sbm_score["dense"] = sbm_test(s,dense_p,gt,args.n_init)
        sbm_scores.append(sbm_score)

    # # SBM Random Graph: 
    # # Variables: 
    # n_classes = 10
    # N1,N2 = 1000,10000
    # sizes1,sizes2 = [N1//n_classes]*n_classes,[N2//n_classes]*n_classes
    # gt1,gt2 = np.array([[val]*(N1//n_classes) for val in range(10)]), np.array([[val]*(N2//n_classes) for val in range(10)])
    # gt1,gt2 = gt1.flatten(),gt2.flatten()
    # # P, edge prob between 2 classes, defines if our matrix is dense or not
    # dense_p = np.array([random.uniform(0,1)]*(n_classes*n_classes))
    # dense_p = np.reshape(dense_p,(n_classes,n_classes))
    # sparse_p1,sparse_p2 = np.array([random.uniform(0,N1)/N1 for _ in range(pow(n_classes,2))]),np.array([random.uniform(0,N2)/N2 for _ in range(pow(n_classes,2))])
    # sparse_p1,sparse_p2 =  np.reshape(sparse_p1,(n_classes,n_classes)), np.reshape(sparse_p2,(n_classes,n_classes))

    # # Dense with N1
    # adj_mat,len_classes = SBMGenerator(sizes1,dense_p)
    # sbmscores_dense1 = [] 
    # sbmscores_dense1.append(applySpectral(len_classes,adj_mat,gt1,n_init=N_INIT))
    # sbmscores_dense1.append(applySpectral(len_classes,adj_mat,gt1,assign_labels="discretize")[1])
    # sbmscores_dense1.append(applySpectral(len_classes,adj_mat,gt1,assign_labels="cluster_qr")[1])
    # sbmscores_dense1.append(applyKMeans(len_classes,adj_mat,gt1,N_INIT)[1]) 
    # # Dense with N2
    # adj_mat,len_classes = SBMGenerator(sizes2,dense_p)
    # sbmscores_dense2 = [] 
    # sbmscores_dense2.append(applySpectral(len_classes,adj_mat,gt2,n_init=N_INIT))
    # sbmscores_dense2.append(applySpectral(len_classes,adj_mat,gt2,assign_labels="discretize")[1])
    # sbmscores_dense2.append(applySpectral(len_classes,adj_mat,gt2,assign_labels="cluster_qr")[1])
    # sbmscores_dense2.append(applyKMeans(len_classes,adj_mat,gt2,N_INIT)[1]) 
    # # Sparse with N1
    # adj_mat,len_classes = SBMGenerator(sizes1,sparse_p1)
    # sbmscores_sparse1 = [] 
    # sbmscores_sparse1.append(applySpectral(len_classes,adj_mat,gt1,n_init=N_INIT))
    # sbmscores_sparse1.append(applySpectral(len_classes,adj_mat,gt1,assign_labels="discretize")[1])
    # sbmscores_sparse1.append(applySpectral(len_classes,adj_mat,gt1,assign_labels="cluster_qr")[1])
    # sbmscores_sparse1.append(applyKMeans(len_classes,adj_mat,gt1,N_INIT)[1]) 
    # # Sparse with N2
    # adj_mat,len_classes = SBMGenerator(sizes2,sparse_p2)
    # sbmscores_sparse2 = [] 
    # sbmscores_sparse2.append(applySpectral(len_classes,adj_mat,gt2,n_init=N_INIT))
    # sbmscores_sparse2.append(applySpectral(len_classes,adj_mat,gt2,assign_labels="discretize")[1])
    # sbmscores_sparse2.append(applySpectral(len_classes,adj_mat,gt2,assign_labels="cluster_qr")[1])
    # sbmscores_sparse2.append(applyKMeans(len_classes,adj_mat,gt2,N_INIT)[1]) 

