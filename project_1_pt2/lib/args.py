import argparse

parser = argparse.ArgumentParser(prog='Testing Spectral Clustering Metrics')

# Mixed Utilities Parameters
parser.add_argument('--verbose', metavar='N', default=False,type=lambda x: True if x=="True" else False,
                    help='Print what the script is doing in stdout.')
parser.add_argument('--output_dir', metavar='N', type=str, default="outputs",
                    help='Output directory for storing images and log file.')
parser.add_argument('--plt', metavar='N', default=True,type=lambda x: True if x=="True" else False,
                    help='Plot the generated data')
parser.add_argument('--save_log', metavar='N', default=True,type=lambda x: True if x=="True" else False,
                    help='Decide if it has to save the log file or not')

# Cluster Parameters  
parser.add_argument('--cluster_method', metavar='N', type=str, default="all",choices= ["all","kmeans","discretize","cluster_qr","DBScan",], #"knn"
                    help='Wich clustering methods has to be used on eigen Vectors.')
parser.add_argument('--n_class', metavar='N', type=int, default=20,
                    help='Number of classes the codes tries to infer.')
parser.add_argument('--laplacian', metavar='N', type=str, default="unnormalized",choices=["unnormalized","symmetric","random-walk"],
                    help='Diffrent types of Laplacian Matrix')

# Graph Pruning Parameters
parser.add_argument('--pruning', metavar='N', default=True,type=lambda x: True if x=="True" else False,
                    help='If we prune the graph or not')
parser.add_argument('--pruning_factor', metavar='N', type=int, default=3,
                    help='Factor for the maximum size of the connected components that get pruned')

def args_getter(): return parser.parse_args()
