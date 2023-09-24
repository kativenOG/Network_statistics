from networkx.linalg import laplacian_matrix
from scipy.sparse.linalg import eigs
import numpy as np
import networkx as nx 
import math

def generate_degree(adj_matrix):
    col_vector = np.sum(adj_matrix,axis=0)
    D = np.diag(col_vector)
    return D 

def generate_laplacian(graph,adj_matrix,ltype="unnormalized"):
    laplacian = nx.laplacian_matrix(graph)
    D  = generate_degree(adj_matrix)
    if ltype=="unnormalized": return laplacian 
    elif ltype=="symmetric": 
        inverse_squared_D = D**(-1/2)
        laplacian =  inverse_squared_D*laplacian*inverse_squared_D
    elif ltype=="random-walk": 
        inverse_D= D**(-1)
        laplacian =  inverse_D*laplacian
    return laplacian 

def eigen_problem(laplacian,n_class= 20 ,n_cc=2):
    f_laplacian = laplacian.asfptype()
    N,upper_bound,lower_bound= n_cc + n_class, pow(math.e,-14), -pow(math.e,-14)
    N = 20 
    print(f"Upper Bound: {upper_bound}\nLower Bound: {lower_bound}\n")
    while True:
        print(N)
        if N <= 0: 
            print("There is something wrong!")
            exit()
        vals_disjoint, vecs_disjoint = eigs(f_laplacian,N,which='SR')
        vals_disjoint = np.sort(vals_disjoint,).reshape(len(vals_disjoint),1)
        zeros = len([value  for value in vals_disjoint if lower_bound < value < upper_bound]) 
        print(f"Zeros: {zeros}")
        if zeros <= n_cc+1: return np.real(vals_disjoint), np.real(vecs_disjoint)
        else: N-=5

