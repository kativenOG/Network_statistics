from lib.all import * 
import networkx as nx  
import warnings 

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = args_getter()
    G,adj_mat = ns_generator() 
    n_cc = nx.number_connected_components(G)
    print(f"Number Connected Components: {n_cc}")
    laplacian = generate_laplacian(G,adj_mat)
    vals_disjoint, vecs_disjoint = eigen_problem(laplacian,n_class=args.n_class,n_cc = n_cc)
    print(f"Eigen values{vals_disjoint.shape}:\n{vals_disjoint}\n\nEigen Vectors{vecs_disjoint.shape}:\n{vecs_disjoint}")
