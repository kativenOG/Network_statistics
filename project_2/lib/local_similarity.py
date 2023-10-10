import networkx as nx 

def sort_scores(piter,key=2,reverse=True):
    """
    From an iterator return a list of sorted tuples (u,v,score). 
    The default behavior is to sort the tuples on their scores from highest to lowest.
    The tuples can be sorted on nodes indexes by changing the key param 
    """
    piter = list(piter) 
    return sorted(piter,key=lambda x: x[key],reverse=reverse)

def cn_score(G):
    """
    Common Neighbors score of all non existing edges of graph G. 
    """
    non_edges = nx.non_edges(G)
    cn_score = (tuple([u,v,len(nx.common_neighbors(u,v))]) for u,v in non_edges)
    return sort_scores(cn_score) 


def leight_holme_newman_1(G):
    """
    Assign high similarity to pairs that have a high expected
    number of neighbours.
    
    s_xy = CN(x,y) / (k_x × k_y)

    Where k_x × k_y is proportional to the number of common
    neighbours of node x and y in an instance of the
    Configuration Model
    """
    lhn1_score = []
    non_edges = nx.non_edges(G) 
    for u,v in non_edges:
        denominator = list(nx.preferential_attachment(G,ebunch=[(u,v)]))[0]
        score = nx.common_neighbors(u,v)/denominator
        lhn1_score.append(tuple([u,v,score]))
    return sort_scores(lhn1_score) 

def preferential_attachment_wrapper(G):
    """
    Easiest score to compute
    s_xy = k_x × k_y
    """
    scores = sort_scores(nx.preferential_attachment(G,ebunch=None),reverse=True)
    return scores

def jaccard_wrapper(G):
    """
    Ratio of common_neighbors and the union of the two common_neighborhoods of x and y.
    """
    scores = sort_scores(nx.jaccard_coefficient(G,ebunch=None),reverse=True)
    return scores

def adamic_adar_wrapper(G):
    """
    Sum over the common common_neighbors of the ratio of 1 and the logarithm of the degree of the node 
    """
    scores = sort_scores(nx.adamic_adar_index(G,ebunch=None),reverse=True)
    return scores

def resource_allocation_wrapper(G):
    """
    Sum over the common common_neighbors of the ratio of 1 over and the degree of the node 
    """
    scores = sort_scores(nx.resource_allocation_index(G,ebunch=None),reverse=True)
    return scores
