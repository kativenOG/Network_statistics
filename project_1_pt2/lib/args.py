import argparse

parser = argparse.ArgumentParser(prog='Testing Spectral Clustering Metrics')

parser.add_argument('--cluster', metavar='N', type=str, default="kmeans",choices=["kmeans","discretize","knn","cluster_qr","DBScan"], # ,"ns"],
                    help='Decide which is the right output for the CLI.')
parser.add_argument('--laplacian', metavar='N', type=str, default="unnormalized",choices=["unnormalized","symmetric","random-walk"],
                    help='Decide which is the right output for the CLI.')
parser.add_argument('--verbose', metavar='N', default=False,type=lambda x: True if x=="True" else False,
                    help='Print what the script is doing in stdout.')
parser.add_argument('--output_dir', metavar='N', type=str, default="outputs",
                    help='Output directory for storing images and log file.')
parser.add_argument('--plt', metavar='N', default=True,type=lambda x: True if x=="True" else False,
                    help='Plot the generated data')

def args_getter():
    return parser.parse_args()
