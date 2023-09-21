from lib.all import * 
import networkx as nx  
import warnings 


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    args = args_getter()
    G,adj_mat = ns_generator() 
    laplacian = generate_laplacian(G,adj_mat)
    vals_disjoint, vecs_disjoint = eigen_problem(laplacian)
    print(f"Eigen values{vals_disjoint.shape}:\n{vals_disjoint}\n\nEigen Vectors{vecs_disjoint.shape}:\n{vecs_disjoint}")
    print()
    print(f"Sorted Eigen Values:\n{np.sort(vals_disjoint,).reshape(len(vals_disjoint),1)}")
