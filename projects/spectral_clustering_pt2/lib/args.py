import argparse

parser = argparse.ArgumentParser(prog='Testing Spectral Clustering Metrics')

# Mixed Utilities Parameters
parser.add_argument('--output_dir', metavar='N', type=str, default="outputs",
                    help='Output directory for storing images and log file.')
parser.add_argument('--plt', metavar='N', default=True,type=lambda x: True if x=="True" else False,
                    help='Plot the generated data')
parser.add_argument('--save_log', metavar='N', default=True,type=lambda x: True if x=="True" else False,
                    help='Decide if it has to save the log file or not')

# Cluster Parameters  
parser.add_argument('--cc_analysis', metavar='N', default=True,type=lambda x: True if x=="True" else False,
                    help='do the analys for every single connected component')
parser.add_argument('--eigen_gap', metavar='N', default=-1,type=int,
                    help='Index for the slicing during eigen gap process, if the value is set to -1 the user will be prompted to insert the input via stdin')
parser.add_argument('--cluster_method', metavar='N', type=str, default="all",choices= ["all","kmeans","discretize","cluster_qr","DBScan"], #"knn"
                    help='Wich clustering methods has to be used on eigen Vectors, the options are [all,kmeans,discretize,cluster_qr",DBScan]')
parser.add_argument('--n_class', metavar='N', type=int, default=20,
                    help='Number of classes the codes tries to infer.')
parser.add_argument('--laplacian', metavar='N', type=str, default="unnormalized",choices=["unnormalized","symmetric","random-walk"],
                    help='Choose the Laplacian Matrix between: [unnormalized,symmetric",random-walk] ')

# Graph Pruning Parameters
parser.add_argument('--pruning', metavar='N', default=True,type=lambda x: True if x=="True" else False,
                    help='If we prune the graph or not')
parser.add_argument('--pruning_factor', metavar='N', type=int, default=20,
                    help='Factor for the maximum size of the connected components that get pruned')

def args_getter(): return parser.parse_args()
