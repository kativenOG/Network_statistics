import networkx as nx
import numpy as np
from numpy.linalg import inv

def katz_index(G):
    """
        Variant of the shortest path metric
        Aggregates over all the paths between x and y and dumps exponentially for longer
        paths to penalize them

        S(x,y) = \sum_{l=1}^{\infty} \beta^{l} |paths_{x,y}^{(l)}|
        = \sum_{l=1}^{\infty} \beta^{l}(A^{l})_{x,y}

        where paths_{x,y}^{(l)} is the set of total l length paths between x and y
        and beta is the damping factor that controls the path weights
    """


    L = nx.normalized_laplacian_matrix(G)
    e = np.lingalg.eigvals(L.A)
    beta = 1/max(e)
    I = np.identity(len(G.nodes))
    return inv(I - nx.to_numpy_array(G)*beta)

