import argparse

parser = argparse.ArgumentParser(prog='Testing Spectral Clustering Metrics')

parser.add_argument('--mode', metavar='N', type=str, required=True,choices=["all","kc","sbm"],
                    help='Decide which is the right output for the CLI.')
parser.add_argument('--n_init', metavar='N', type=int,default=100,
                    help='Number of iterations for k_means.')
parser.add_argument('--n_vals', metavar='N', type=int,default=100,nargs="+",
                    help='Number of vertices for SBM.')
parser.add_argument('--n_classes', metavar='N', type=int,default=10,
                    help='Number of community classes for SBM.')
parser.add_argument('--output_dir', metavar='N', type=str, default="outputs",
                    help='Output directory for storing images and csv.')

def args_getter():
    return parser.parse_args()
