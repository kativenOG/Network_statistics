import argparse

parser = argparse.ArgumentParser(prog='Testing Spectral Clustering Metrics')

parser.add_argument('--cluster', metavar='N', type=str, default="all",choices= ["all","kmeans","discretize","cluster_qr","DBScan",], #"knn"
                    help='Wich clustering methods has to be used on eigen Vectors.')
parser.add_argument('--n_class', metavar='N', type=int, default=20,
                    help='Number of classes the codes tries to infer.')
parser.add_argument('--laplacian', metavar='N', type=str, default="unnormalized",choices=["unnormalized","symmetric","random-walk"],
                    help='Diffrent types of Laplacian Matrix')
parser.add_argument('--verbose', metavar='N', default=False,type=lambda x: True if x=="True" else False,
                    help='Print what the script is doing in stdout.')
parser.add_argument('--output_dir', metavar='N', type=str, default="outputs",
                    help='Output directory for storing images and log file.')
parser.add_argument('--plt', metavar='N', default=True,type=lambda x: True if x=="True" else False,
                    help='Plot the generated data')

def args_getter():
    return parser.parse_args()
