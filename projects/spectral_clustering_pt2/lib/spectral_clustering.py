import numpy as np
import networkx as nx 
from scipy.sparse.linalg import eigs

def generate_degree(adj_matrix):
    col_vector = np.sum(adj_matrix,axis=0)
    D = np.diag(col_vector)
    return D 

def generate_laplacian(graph,ltype="unnormalized"):
    print(f"The program is using the {ltype} laplacian\n")
    laplacian = nx.laplacian_matrix(graph)
    adj_mat = nx.to_numpy_array(graph)
    D  = generate_degree(adj_mat)
    if   ltype == "unnormalized": return laplacian 
    elif ltype=="symmetric": 
        inverse_squared_D = D**(-1/2)
        laplacian =  inverse_squared_D*laplacian*inverse_squared_D
    elif ltype=="random-walk": 
        inverse_D= D**(-1)
        laplacian =  inverse_D*laplacian
    print(laplacian)
    return laplacian 

def eigen_problem(laplacian,n_class= 20 ,n_cc=2,eigen_gap =-1):
    print("Solving the Eigen_problem")
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
        
