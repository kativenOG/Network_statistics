import numpy as np
import networkx as nx 
import math
from scipy.sparse.linalg import eigs
import scipy as sp
import scipy.sparse

def generate_degree(adj_matrix):
    col_vector = np.sum(adj_matrix,axis=0)
    D = np.diag(col_vector)
    return D 

def generate_laplacian(graph,adj_matrix,ltype="unnormalized"):
    laplacian = nx.laplacian_matrix(graph)
    D  = generate_degree(adj_matrix)
    if   ltype == "unnormalized": return laplacian 
    elif ltype=="symmetric": 
        inverse_squared_D = D**(-1/2)
        laplacian =  inverse_squared_D*laplacian*inverse_squared_D
    elif ltype=="random-walk":
        # Source code from NetworkX is used for creating normalized graph
        nodelist = list(graph)
        A = nx.to_scipy_sparse_array(graph, nodelist=nodelist, weight='weight', format="csr")
        n, m = A.shape
        diags = A.sum(axis=1)
        D = sp.sparse.csr_array(sp.sparse.spdiags(diags, 0, m, n, format="csr"))
        with sp.errstate(divide="ignore"):
            diags_inverse = 1.0 / np.sqrt(diags)
        diags_inverse[np.isinf(diags_inverse)] = 0
        ID = sp.sparse.csr_array(sp.sparse.spdiags(diags_inverse, 0, m, n, format="csr"))
        laplacian =  ID @ laplacian
    return laplacian 

def eigen_problem(laplacian,n_class= 20 ,n_cc=2,eigen_gap =-1):
    f_laplacian = laplacian.asfptype()
    N,upper_bound,lower_bound= n_cc + n_class, pow(1,-14), -pow(1,-14)
    print(f"Upper Bound: {upper_bound}\nLower Bound: {lower_bound}\n")
    vals_disjoint, vecs_disjoint = eigs(f_laplacian,N,which='SR')
    vals_disjoint = np.sort(vals_disjoint,).reshape(len(vals_disjoint),1)
    vals_disjoint, vecs_disjoint = np.real(vals_disjoint), np.real(vecs_disjoint) # remove the immaginary part  

    # Number of zeroes 
    zeros = len([value  for value in vals_disjoint if lower_bound < value < upper_bound]) 
    if zeros == n_cc: print(f"We have the right number of Zeros ({n_cc})!")
    else: print(f"We dont't have the right number of Zeros ({zeros}/{n_cc})! ")

    # Eigen Gap 
    if eigen_gap == -1:
        print("apply eigen Gap: ".upper())
        print("Eigen Values:")
        print("\n".join(list(map(lambda x : f"-{x[0]}: {x[1]} ",enumerate(vals_disjoint.flatten())))))
        slice_line = ""
        while not slice_line.isdigit(): slice_line = input("Insert the line where the eigen values need to be sliced: ")
        slice_line = int(slice_line)
        vecs_disjoint = vecs_disjoint [:,:slice_line]
    else: vecs_disjoint = vecs_disjoint [:,:eigen_gap] 

    return vals_disjoint,vecs_disjoint 
        
