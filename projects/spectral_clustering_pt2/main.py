import warnings,os 
from lib.all import * 
args = args_getter()

def main(G,n_cc,n_class,component_id=""):
    # Do the process for the whole graph or just a single view (connected component) of it   
    laplacian = generate_laplacian(G,args.laplacian)
    # Solving the Eigen Problem 
    vals_disjoint, vecs_disjoint = eigen_problem(laplacian,n_class=n_class,n_cc = n_cc,eigen_gap=args.eigen_gap)
    print(f"Eigen values{vals_disjoint.shape}:\n{vals_disjoint}\n\nEigen Vectors{vecs_disjoint.shape}:\n{vecs_disjoint}")

    # Clustering the Eigen Vectors
    c_vector= vector_clustering(vecs_disjoint,args.cluster_method,epsilon=0.5) 
    results = c_vector.cluster()
    results_shapes = [result.shape for result in results.values()]
    print(f"Results Shape: {results_shapes}")
    for exp,result in results.items(): print(f"In {exp} the result was: {result}")
    
    # Save Logs of counter in classes  
    save_counter_log(results,full_path,args.save_log,component_id)
    # Plots and Metrics:
    if args.plt:
        print("Plot of the Initial Graph (no clustering)")
        title = os.path.join(full_path,"starting_plot.png")
        draw_network(G,title)
        for clustering_method,result in results.items():
            print(f"Plot of the Graph using spectral clustering and {clustering_method}!")
            title = os.path.join(full_path,str(clustering_method.upper() + component_id + "_plot.png"))
            draw_network(G,title,gt=result)
            



if __name__ == "__main__":
    # STD stuff 
    warnings.filterwarnings("ignore") 
    full_path = os.path.join(os.getcwd(),args.output_dir)
    if os.path.isdir(full_path):  
        shutil.rmtree(full_path)
    os.mkdir(full_path) 

    # Get the graph and generate the Laplacian Matrix 
    G,adj_mat = ns_generator(args.pruning,args.pruning_factor) 
    ccs,n_cc = ccn_wrapper(G)
    # ccs = sorted(ccs,key=lambda x: len(x),reverse=True)[:args.pruning_factor] # should be useless
    print(f"Number of Connected Components: {n_cc}")
    if not args.cc_analysis: main(G,n_cc,args.n_class)
    elif args.pruning : # Doing the same stuff with Subgraphs if pruning in activated 
        print("Analyzing Every Connected Component!")
        for i,cc in enumerate(ccs): 
            print(f"Connected component number {i+1} of lenght {len(cc)}")
            n_class= ""
            while not n_class.isdigit(): n_class = input("Insert the number of classes (eigen values): ")
            view,v_adj  = generating_graph_view(G,cc)
            main(view,1,int(n_class),f"_component_{i+1}_")
