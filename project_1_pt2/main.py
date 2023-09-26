import warnings,os 
from lib.all import * 

if __name__ == "__main__":
    # STD stuff 
    warnings.filterwarnings("ignore") 
    args = args_getter()
    full_path = os.path.join(os.getcwd(),args.output_dir)
    if os.path.isdir(full_path):  
        shutil.rmtree(full_path)
    os.mkdir(full_path) 

    # Get the graph and generate the Laplacian Matrix 
    G,adj_mat = ns_generator(args.pruning,args.pruning_factor) 
    n_cc = ccn_wrapper(G)
    if args.verbose: print(f"Number Connected Components: {n_cc}")
    laplacian = generate_laplacian(G,adj_mat)
    
    # Solving the Eigen Problem 
    vals_disjoint, vecs_disjoint = eigen_problem(laplacian,n_class=args.n_class,n_cc = n_cc,eigen_gap=args.eigen_gap)
    if args.verbose: print(f"Eigen values{vals_disjoint.shape}:\n{vals_disjoint}\n\nEigen Vectors{vecs_disjoint.shape}:\n{vecs_disjoint}")

    # Clustering the Eigen Vectors
    vector_clustering = vector_clustering(vecs_disjoint,args.cluster_method,n_clusters= args.n_class,epsilon = 3) 
    results = vector_clustering.cluster()
    results_shapes = [result.shape for result in results.values()]
    if args.verbose: 
        print(f"Results Shape: {results_shapes}")
        for exp,result in results.items(): print(f"In {exp} the result was: {result}")
    
    # Save Logs of counter in classes  
    save_counter_log(results,args.verbose,full_path,args.save_log)
    # Plots and Metrics:
    if args.plt:
        if args.verbose: print("Plot of the Initial Graph (no clustering)")
        title = os.path.join(full_path,"starting_plot.png")
        draw_network(G,title)
        for clustering_method,result in results.items():
            if args.verbose: print(f"Plot of the Graph using spectral clustering and {clustering_method}!")
            title = os.path.join(full_path,str(clustering_method.upper() + "_plot.png"))
            draw_network(G,title,gt=result)

    scores = compute_performance(G, results)
    print(scores)