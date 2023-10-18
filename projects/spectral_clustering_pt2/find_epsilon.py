from numpy import full
from lib.all import * 
import warnings,os 

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = args_getter()
    full_path = os.path.join(os.getcwd(),args.output_dir)
    G,adj_mat = ns_generator(args.pruning,args.pruning_factor) 
    n_cc = ccn_wrapper(G)
    laplacian = generate_laplacian(G,adj_mat)
    vals_disjoint, vecs_disjoint = eigen_problem(laplacian,n_class=args.n_class,n_cc = n_cc,eigen_gap=args.eigen_gap)
    vector_clustering = vector_clustering(vecs_disjoint,args.cluster_method,n_clusters= args.n_class,epsilon = 3) 
    vector_clustering.plot_distance_for_epsilon(full_path) 
