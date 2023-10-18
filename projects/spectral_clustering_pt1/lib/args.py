import argparse

parser = argparse.ArgumentParser(prog='Testing Spectral Clustering Metrics')

parser.add_argument('--mode', metavar='N', type=str, default="all",choices=["all","kc","sbm"], # ,"ns"],
                    help='Decide which is the right output for the CLI.')
parser.add_argument('--verbose', metavar='N', default=False,type=lambda x: True if x=="True" else False,
                    help='Print what the script is doing in stdout.')
parser.add_argument('--n_init', metavar='N', type=int,default=100,
                    help='Number of iterations for k_means.')
parser.add_argument('--save_log', metavar='N', default=False,type=lambda x: True if x=="True" else False,
                    help='Create a file containing all of the logs.')
parser.add_argument('--plt', metavar='N', default=True,type=lambda x: True if x=="True" else False,
                    help='Plot the generated data')
parser.add_argument('--n_vals', metavar='N', type=int,default=100,nargs="+",
                    help='Number of vertices for SBM.')
parser.add_argument('--n_classes', metavar='N', type=int,default=10,
                    help='Number of community classes for SBM.')
parser.add_argument('--output_dir', metavar='N', type=str, default="outputs",
                    help='Output directory for storing images and log file.')

def args_getter():
    return parser.parse_args()
