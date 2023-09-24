import numpy as np 
from scipy.linalg import LinAlgError, qr, svd
from sklearn.metrics import DistanceMetric 
from sklearn.utils import as_float_array, check_random_state
from scipy.sparse import csc_matrix
from sklearn.cluster import KMeans,DBSCAN
from itertools import product
import matplotlib.pyplot as plt

class vector_clustering(): 

    def __init__(self,eigen_vecs,methods,n_clusters,epsilon=3) -> None:
        if methods == "all": self.methods = ["kmeans","discretize","cluster_qr","DBScan"]
        else: self.methods = [methods]
        self.n_clusters = n_clusters
        self.vectors = eigen_vecs 
        self.epsilon = epsilon 
        self.minPts  = len(eigen_vecs[0]) + 1 # Usually the number of dimensions (eigen_values) + 1 is used as a starting point
        if self.minPts < 3: self.minPts = 3   # The value shouldn't be lower than 3 
 
    def cluster(self) -> dict:
        """
        Normal main function for the class, used to run all the different clustering algorithms 
        given in input trough command line.
        """
        result_labels = {}
        for method in self.methods :
            func = getattr(self, method)
            result_labels[method] = func()
        return result_labels 

    def plot_distance_for_epsilon(self): 
        """
        Other function for the class, used to plot the distance matrix to compute the epsilon 
        for the DBSCAN algorithm.
        """
        # create distance Matrix 
        distance_matrix, distance_metric = [], DistanceMetric().get_metric("euclidean")
        x_shape,y_shape = list(self.vectors[0].shape),list(self.vectors[0].shape)
        x_shape.insert(0,1)
        y_shape.insert(0,1)
        x_shape,y_shape = tuple(x_shape),tuple(y_shape)
        for x,y in product(self.vectors,self.vectors):
            x,y = np.reshape(x,x_shape),np.reshape(y,y_shape)
            euclidian_distance = distance_metric.pairwise(x,y)
            distance_matrix.append(euclidian_distance[0])
        print("Done Calculating")

        # Plot:
        SHAPE =  tuple([len(self.vectors),len(self.vectors)])
        distance_matrix = np.reshape(np.array(distance_matrix),SHAPE)
        plt.plot(list(range(0,SHAPE[0])),distance_matrix[:,self.minPts],"r")
        plt.plot(list(range(0,SHAPE[0])),distance_matrix[self.minPts,:],"b")
        plt.show() 
        plt.savefig("distance_graph.png")

    def kmeans(self):
        clustering = KMeans(n_clusters=self.n_clusters).fit(self.vectors) 
        return clustering.labels_

    def DBScan(self):
        """
        Parameters: 
            - epsilon: Maximum radius for wich points are considered neighbors of each other
            - minPts:  Minumum number of nodes needed to create a new cluster (has to be greater than 3).
        Main algorithm:
         We start with the data points and values of epsilon and minPts as input:

           1- We select a random starting point that has not been visited.
           2- Determine the neighborhood of this point using epsilon which essentially acts as a radius.
           3- If the points in the neighborhood satisfy minPts criteria then the point is marked as a core point. The clustering process will start and the point is marked as visited else this point is labeled as noise.
           4- All points within the neighborhood of the core point are also marked as part of the cluster and the above procedure from step 2 is repeated for all epsilon neighborhood points.
           5- A new unvisited point is fetched, and following the above steps they are either included to form another cluster or they are marked as noise.
           6- The above process is continued till all points are visited.
        """
        clustering = DBSCAN(eps=self.epsilon, min_samples=self.minPts).fit(self.vectors)
        return clustering.labels_

    def cluster_qr(self):
        """
        Find the discrete partition closest to the eigenvector embedding.
    
        This implementation was proposed in [1]_.
        .. versionadded:: 1.1
    
        Parameters
        ----------
        vectors : array-like, shape: (n_samples, n_clusters)
            The embedding space of the samples.
    
        Returns
        -------
        labels : array of integers, shape: n_samples
            The cluster labels of vectors.
    
        References
        ----------
        .. [1] :doi:`Simple, direct, and efficient multi-way spectral clustering, 2019
            Anil Damle, Victor Minden, Lexing Ying
            <10.1093/imaiai/iay008>`
    
        """
    
        k = self.vectors.shape[1]
        _, _, piv = qr(self.vectors.T, pivoting=True)
        ut, _, v = svd(self.vectors[piv[:k], :].T)
        vectors = abs(np.dot(self.vectors, np.dot(ut, v.conj())))
        return vectors.argmax(axis=1)
    
    
    def discretize(self, copy=True, max_svd_restarts=30, n_iter_max=20, random_state=None):
        """Search for a partition matrix which is closest to the eigenvector embedding.
    
        This implementation was proposed in [1]_.
    
        Parameters
        ----------
        vectors : array-like of shape (n_samples, n_clusters)
            The embedding space of the samples.
    
        copy : bool, default=True
            Whether to copy vectors, or perform in-place normalization.
    
        max_svd_restarts : int, default=30
            Maximum number of attempts to restart SVD if convergence fails
    
        n_iter_max : int, default=30
            Maximum number of iterations to attempt in rotation and partition
            matrix search if machine precision convergence is not reached
    
        random_state : int, RandomState instance, default=None
            Determines random number generation for rotation matrix initialization.
            Use an int to make the randomness deterministic.
            See :term:`Glossary <random_state>`.
    
        Returns
        -------
        labels : array of integers, shape: n_samples
            The labels of the clusters.
    
        References
        ----------
    
        .. [1] `Multiclass spectral clustering, 2003
               Stella X. Yu, Jianbo Shi
               <https://people.eecs.berkeley.edu/~jordan/courses/281B-spring04/readings/yu-shi.pdf>`_
    
        Notes
        -----
    
        The eigenvector embedding is used to iteratively search for the
        closest discrete partition.  First, the eigenvector embedding is
        normalized to the space of partition matrices. An optimal discrete
        partition matrix closest to this normalized embedding multiplied by
        an initial rotation is calculated.  Fixing this discrete partition
        matrix, an optimal rotation matrix is calculated.  These two
        calculations are performed until convergence.  The discrete partition
        matrix is returned as the clustering solution.  Used in spectral
        clustering, this method tends to be faster and more robust to random
        initialization than k-means.
    
        """
    
        random_state = check_random_state(random_state)
    
        vectors = as_float_array(self.vectors, copy=copy)
    
        eps = np.finfo(float).eps
        n_samples, n_components = vectors.shape
    
        # Normalize the eigenvectors to an equal length of a vector of ones.
        # Reorient the eigenvectors to point in the negative direction with respect
        # to the first element.  This may have to do with constraining the
        # eigenvectors to lie in a specific quadrant to make the discretization
        # search easier.
        norm_ones = np.sqrt(n_samples)
        for i in range(vectors.shape[1]):
            vectors[:, i] = (vectors[:, i] / np.linalg.norm(vectors[:, i])) * norm_ones
            if vectors[0, i] != 0:
                vectors[:, i] = -1 * vectors[:, i] * np.sign(vectors[0, i])
    
        # Normalize the rows of the eigenvectors.  Samples should lie on the unit
        # hypersphere centered at the origin.  This transforms the samples in the
        # embedding space to the space of partition matrices.
        vectors = vectors / np.sqrt((vectors**2).sum(axis=1))[:, np.newaxis]
    
        svd_restarts = 0
        has_converged = False
    
        # If there is an exception we try to randomize and rerun SVD again
        # do this max_svd_restarts times.
        while (svd_restarts < max_svd_restarts) and not has_converged:
            # Initialize first column of rotation matrix with a row of the
            # eigenvectors
            rotation = np.zeros((n_components, n_components))
            rotation[:, 0] = vectors[random_state.randint(n_samples), :].T
    
            # To initialize the rest of the rotation matrix, find the rows
            # of the eigenvectors that are as orthogonal to each other as
            # possible
            c = np.zeros(n_samples)
            for j in range(1, n_components):
                # Accumulate c to ensure row is as orthogonal as possible to
                # previous picks as well as current one
                c += np.abs(np.dot(vectors, rotation[:, j - 1]))
                rotation[:, j] = vectors[c.argmin(), :].T
    
            last_objective_value = 0.0
            n_iter = 0
    
            while not has_converged:
                n_iter += 1
    
                t_discrete = np.dot(vectors, rotation)
    
                labels = t_discrete.argmax(axis=1)
                vectors_discrete = csc_matrix(
                    (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
                    shape=(n_samples, n_components),
                )
    
                t_svd = vectors_discrete.T * vectors
    
                try:
                    U, S, Vh = np.linalg.svd(t_svd)
                except LinAlgError:
                    svd_restarts += 1
                    print("SVD did not converge, randomizing and trying again")
                    break
    
                ncut_value = 2.0 * (n_samples - S.sum())
                if (abs(ncut_value - last_objective_value) < eps) or (n_iter > n_iter_max):
                    has_converged = True
                else:
                    # otherwise calculate rotation and continue
                    last_objective_value = ncut_value
                    rotation = np.dot(Vh.T, U.T)
    
        if not has_converged:
            raise LinAlgError("SVD did not converge")
        return labels
