import numpy as np
from sklearn.neighbors import KDTree


def chamfer(pc1, pc2, p=2):
    """
    Computes the chamfer distance between two point clouds.

    pc1 (numpy.ndarray): array with shape [N, 3] with the points of one point cloud
    pc2 (numpy.ndarray): array with shape [M, 3] with the points of other point cloud
    p (float): the order of the distance to use for the points. 1 <= p <= infinity
    """
    # Distance of pc2 to pc1
    tree1 = KDTree(pc1)
    dists1, _ind1 = tree1.query(pc2)
    dist1 = np.mean(dists1)

    # Distance of pc1 to pc2
    tree2 = KDTree(pc2)
    dists2, _ind2 = tree2.query(pc1)
    dist2 = np.mean(dists2)

    return dist1 + dist2
