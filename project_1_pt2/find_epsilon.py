from lib.all import * 
import networkx as nx  
import warnings 

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = args_getter()
    G,adj_mat = ns_generator() 
    n_cc = nx.number_connected_components(G)
    laplacian = generate_laplacian(G,adj_mat)
    vals_disjoint, vecs_disjoint = eigen_problem(laplacian,n_class=args.n_class,n_cc = n_cc)
    vector_clustering = vector_clustering(vecs_disjoint,args.cluster_method,n_clusters= args.n_class,epsilon = 3) 
    vector_clustering.plot_distance_for_epsilon() 
