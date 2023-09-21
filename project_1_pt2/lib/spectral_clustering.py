from networkx.linalg import laplacian_matrix
from scipy.sparse.linalg import eigs
import numpy as np
import networkx as nx 

import os,shutil,subprocess

def ns_generator():
    # Get rid of old repositories if present (just to be sure)
    dir1,dir2  =  os.path.join(os.getcwd(),"{}"),os.path.join(os.getcwd(),"dimacs10-netscience")
    if os.path.isdir(dir1): shutil.rmtree(dir1)
    if os.path.isdir(dir2): shutil.rmtree(dir2)

    # Download the dataset and unzip it 
    subprocess.run(["wget","-P","{}","http://konect.cc/files/download.tsv.dimacs10-netscience.tar.bz2"])
    subprocess.run(["bzip2","-d","{}/download.tsv.dimacs10-netscience.tar.bz2"])
    subprocess.run(["tar","-xf","{}/download.tsv.dimacs10-netscience.tar"])
     
    # Remove the first line from the Adjecency file
    with open("dimacs10-netscience/out.dimacs10-netscience","r") as f:
        lines = f.readlines()        
    with open("dimacs10-netscience/out.dimacs10-netscience","w") as f:
        for line in lines[1:]: f.write(line)

    # Create a graph and get rid of the dataset files
    G =  nx.read_adjlist("dimacs10-netscience/out.dimacs10-netscience")
    subprocess.run(["rm","-rf","{}","dimacs10-netscience"])
    adj_mat = nx.to_numpy_array(G) # Return the graph adjacency matrix as a NumPy matrix.

    return G,adj_mat


def generate_degree(adj_matrix):
    col_vector = np.sum(adj_matrix,axis=0)
    D = np.diag(col_vector)
    print(D)
    print(D.shape)
    return D 

def generate_laplacian(graph,adj_matrix,ltype="unnormalized"):
    laplacian = nx.laplacian_matrix(graph)
    D  = generate_degree(adj_matrix)
    if ltype=="unnormalized": return laplacian 
    elif ltype=="symmetric": 
        inverse_squared_D = D**(-1/2)
        laplacian =  inverse_squared_D*laplacian*inverse_squared_D
    elif ltype=="random-walk": 
        inverse_D= D**(-1)
        laplacian =  inverse_D*laplacian
    return laplacian 

def eigen_problem(laplacian):
    f_laplacian = laplacian.asfptype()
    N = 50 
    print(N)
    vals_disjoint, vecs_disjoint = eigs(f_laplacian,N,which='SR')
    return np.real(vals_disjoint), np.real(vecs_disjoint)



# from sklearn.base import BaseEstimator, ClusterMixin, _fit_context
# from scipy.sparse import csc_matrix
# from sklearn.manifold import spectral_embedding
# from sklearn.metrics.pairwise import KERNEL_PARAMS, pairwise_kernels
# from sklearn.neighbors import NearestNeighbors, kneighbors_graph
# from sklearn.utils import as_float_array, check_random_state
# from sklearn.cluster._kmeans import k_means
# from scipy.linalg import LinAlgError, qr, svd

# def cluster_qr(vectors):
#     """Find the discrete partition closest to the eigenvector embedding.

#         This implementation was proposed in [1]_.
# .. versionadded:: 1.1

#         Parameters
#         ----------
#         vectors : array-like, shape: (n_samples, n_clusters)
#             The embedding space of the samples.

#         Returns
#         -------
#         labels : array of integers, shape: n_samples
#             The cluster labels of vectors.

#         References
#         ----------
#         .. [1] :doi:`Simple, direct, and efficient multi-way spectral clustering, 2019
#             Anil Damle, Victor Minden, Lexing Ying
#             <10.1093/imaiai/iay008>`

#     """

#     k = vectors.shape[1]
#     _, _, piv = qr(vectors.T, pivoting=True)
#     ut, _, v = svd(vectors[piv[:k], :].T)
#     vectors = abs(np.dot(vectors, np.dot(ut, v.conj())))
#     return vectors.argmax(axis=1)


# def discretize(vectors, *, copy=True, max_svd_restarts=30, n_iter_max=20, random_state=None):
#     """Search for a partition matrix which is closest to the eigenvector embedding.

#     This implementation was proposed in [1]_.

#     Parameters
#     ----------
#     vectors : array-like of shape (n_samples, n_clusters)
#         The embedding space of the samples.

#     copy : bool, default=True
#         Whether to copy vectors, or perform in-place normalization.

#     max_svd_restarts : int, default=30
#         Maximum number of attempts to restart SVD if convergence fails

#     n_iter_max : int, default=30
#         Maximum number of iterations to attempt in rotation and partition
#         matrix search if machine precision convergence is not reached

#     random_state : int, RandomState instance, default=None
#         Determines random number generation for rotation matrix initialization.
#         Use an int to make the randomness deterministic.
#         See :term:`Glossary <random_state>`.

#     Returns
#     -------
#     labels : array of integers, shape: n_samples
#         The labels of the clusters.

#     References
#     ----------

#     .. [1] `Multiclass spectral clustering, 2003
#            Stella X. Yu, Jianbo Shi
#            <https://people.eecs.berkeley.edu/~jordan/courses/281B-spring04/readings/yu-shi.pdf>`_

#     Notes
#     -----

#     The eigenvector embedding is used to iteratively search for the
#     closest discrete partition.  First, the eigenvector embedding is
#     normalized to the space of partition matrices. An optimal discrete
#     partition matrix closest to this normalized embedding multiplied by
#     an initial rotation is calculated.  Fixing this discrete partition
#     matrix, an optimal rotation matrix is calculated.  These two
#     calculations are performed until convergence.  The discrete partition
#     matrix is returned as the clustering solution.  Used in spectral
#     clustering, this method tends to be faster and more robust to random
#     initialization than k-means.

#     """

#     random_state = check_random_state(random_state)

#     vectors = as_float_array(vectors, copy=copy)

#     eps = np.finfo(float).eps
#     n_samples, n_components = vectors.shape

#     # Normalize the eigenvectors to an equal length of a vector of ones.
#     # Reorient the eigenvectors to point in the negative direction with respect
#     # to the first element.  This may have to do with constraining the
#     # eigenvectors to lie in a specific quadrant to make the discretization
#     # search easier.
#     norm_ones = np.sqrt(n_samples)
#     for i in range(vectors.shape[1]):
#         vectors[:, i] = (vectors[:, i] / np.linalg.norm(vectors[:, i])) * norm_ones
#         if vectors[0, i] != 0:
#             vectors[:, i] = -1 * vectors[:, i] * np.sign(vectors[0, i])

#     # Normalize the rows of the eigenvectors.  Samples should lie on the unit
#     # hypersphere centered at the origin.  This transforms the samples in the
#     # embedding space to the space of partition matrices.
#     vectors = vectors / np.sqrt((vectors**2).sum(axis=1))[:, np.newaxis]

#     svd_restarts = 0
#     has_converged = False

#     # If there is an exception we try to randomize and rerun SVD again
#     # do this max_svd_restarts times.
#     while (svd_restarts < max_svd_restarts) and not has_converged:
#         # Initialize first column of rotation matrix with a row of the
#         # eigenvectors
#         rotation = np.zeros((n_components, n_components))
#         rotation[:, 0] = vectors[random_state.randint(n_samples), :].T

#         # To initialize the rest of the rotation matrix, find the rows
#         # of the eigenvectors that are as orthogonal to each other as
#         # possible
#         c = np.zeros(n_samples)
#         for j in range(1, n_components):
#             # Accumulate c to ensure row is as orthogonal as possible to
#             # previous picks as well as current one
#             c += np.abs(np.dot(vectors, rotation[:, j - 1]))
#             rotation[:, j] = vectors[c.argmin(), :].T

#         last_objective_value = 0.0
#         n_iter = 0

#         while not has_converged:
#             n_iter += 1

#             t_discrete = np.dot(vectors, rotation)

#             labels = t_discrete.argmax(axis=1)
#             vectors_discrete = csc_matrix(
#                 (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
#                 shape=(n_samples, n_components),
#             )

#             t_svd = vectors_discrete.T * vectors

#             try:
#                 U, S, Vh = np.linalg.svd(t_svd)
#             except LinAlgError:
#                 svd_restarts += 1
#                 print("SVD did not converge, randomizing and trying again")
#                 break

#             ncut_value = 2.0 * (n_samples - S.sum())
#             if (abs(ncut_value - last_objective_value) < eps) or (n_iter > n_iter_max):
#                 has_converged = True
#             else:
#                 # otherwise calculate rotation and continue
#                 last_objective_value = ncut_value
#                 rotation = np.dot(Vh.T, U.T)

#     if not has_converged:
#         raise LinAlgError("SVD did not converge")
#     return labels

