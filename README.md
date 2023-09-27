# Network_statistics
Repo for notebooks / project for the 2AMS30 Network Statistics for Data Science course.
## Spectral Clustering:
## First Presentation:
```
cd deps
bash installer.sh
cd ..
source networkStats/bin/activate
```
## First Presentation:
### Usage: 
Testing Spectral Clustering Metrics [-h] [--mode N] [--verbose N] [--n_init N] [--save_log N] [--plt N] [--n_vals N [N ...]] [--n_classes N] [--output_dir N]

options:
  -h, --help          show this help message and exit
  --mode N            Decide which is the right output for the CLI.
  --verbose N         Print what the script is doing in stdout.
  --n_init N          Number of iterations for k_means.
  --save_log N        Create a file containing all of the logs.
  --plt N             Plot the generated data
  --n_vals N [N ...]  Number of vertices for SBM.
  --n_classes N       Number of community classes for SBM.
  --output_dir N      Output directory for storing images and log file.

### To run the config used to plot the graphs used in the Presentation type:
```
cd project_1_pt1
python main.py --mode "all" --save_log True --plt True --verbose True --n_vals 10 100 1000
```

## Second Presentation:
### Usage: 
Testing Spectral Clustering Metrics [-h] [--output_dir N] [--plt N] [--save_log N] [--cc_analysis N] [--eigen_gap N] [--cluster_method N] [--n_class N] [--laplacian N] [--pruning N] [--pruning_factor N]

options:
  -h, --help          show this help message and exit
  --output_dir N      Output directory for storing images and log file.
  --plt N             Plot the generated data
  --save_log N        Decide if it has to save the log file or not
  --cc_analysis N     do the analys for every single connected component
  --eigen_gap N       Index for the slicing during eigen gap process, if the value is set to -1 the user will be prompted to insert the input via stdin
  --cluster_method N  Wich clustering methods has to be used on eigen Vectors.
  --n_class N         Number of classes the codes tries to infer.
  --laplacian N       Diffrent types of Laplacian Matrix
  --pruning N         If we prune the graph or not
  --pruning_factor N  Factor for the maximum size of the connected components that get pruned

### To run the clustering algorithm on each connected component type with n>20 type: 
```
cd project_1_pt2
python main.py 
```
### To run the clustering algorithm on the whole pruned graph (still with n>20) type: 
```
cd project_1_pt2
python main.py --cc_analysis False
```
