from networkx import adjacency_matrix
from lib.all import *
import os,shutil,warnings

if __name__ == "__main__":
    # Ignore warnings 
    warnings.filterwarnings("ignore") 

    args = args_getter()
    # Lets get rid of old output repositories and create a new one 
    if os.path.isdir(args.output_dir):  
        shutil.rmtree(os.path.join(os.getcwd(),args.output_dir))
    os.mkdir(os.path.join(os.getcwd(),args.output_dir))

    # Setting up the mode for the CLI 
    # ns_flag = False
    karate_flag,sbm_flag= False, False
    if args.mode == "all": karate_flag,sbm_flag= True,True # ns_flag=True
    elif args.mode == "sbm": sbm_flag = True
    elif args.mode == "kc": karate_flag = True
    # elif args.mode == "ns": ns_flag= True

    # Checking if we got multiple values from command line in the n_vals argument  
    if not isinstance(args.n_vals,list): n_vals = [args.n_vals]
    else: n_vals = args.n_vals

    # Running Simulations:
    k_scores, sbm_scores = None, []
    if karate_flag: 
        k_scores = np.array(karate_club_test(args.n_init,args.verbose))

    if sbm_flag: 
        sizes,sparse_ps,gts,dense_p =generateSBMParams(args.n_classes,n_vals)
        sbm_score = {} 
        for  s,sparse_p,gt in zip(sizes,sparse_ps,gts):
            sbm_score["sparse"] = sbm_test(s,sparse_p,gt,args.n_init,args.verbose,sparse=True)
            sbm_score["dense"] = sbm_test(s,dense_p,gt,args.n_init,args.verbose,sparse=False)
            sbm_scores.append(sbm_score)

    # if ns_flag:
    #     G,adj = ns_generator() 
         

    # Saving the logs 
    if args.save_log == True: 
        log_file = open(os.path.join(args.output_dir,"log_file.txt"),"w")
        if karate_flag: log_file.write(f"Karate Club Graph Data:\n{k_scores}\n")
        if sbm_flag: 
            log_file.write(f"Stocastic Block Model Graph Data:\n")
            for n,s in zip(n_vals,sbm_scores): log_file.write(f"n={n}:\n{s}\n")
        log_file.close()

    # Plotting the Data 
    if args.plt == True: 
        if args.verbose: print("PLOTTING:")
        score_names = ["AMI and MI","ARI and RI"]
        x_labels = ["S-kmeans","S-discretize","S-cluster_qr","kmeans"]
        if karate_flag:
            if args.verbose: print("Karate Club")
            for i,names in enumerate(score_names):
                plot_scores("Karate Club\n"+names,x_labels,k_scores[:,i],args.output_dir,"karate_"+names+".png") 
        if sbm_flag:
            for n,vals in zip(n_vals,sbm_scores):
                if args.verbose: print(f"SBM with N= {n}")
                vals["dense"],vals["sparse"] = np.array(vals["dense"]), np.array(vals["sparse"])
                for i,names in enumerate(score_names):
                    if args.verbose: print("DENSE GRAPH")
                    plot_scores(f"SBM Dense with n={n}\n"+names,x_labels,vals["dense"][:,i],args.output_dir,"sbm_"+f"dense_{str(n)}_"+names+".png") 
                    if args.verbose: print("SPARSE GRAPH")
                    plot_scores(f"SBM Sparse with n={n}\n"+names,x_labels,vals["sparse"][:,i],args.output_dir,"sbm_"+f"sparse_{str(n)}_"+names+".png") 
         
