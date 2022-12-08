from typing import List, Tuple
import numpy as np
from utils import problem
from collections import Counter


@problem.tag("hw4-A")
def calculate_centers(
    data: np.ndarray, classifications: np.ndarray, num_centers: int
) -> np.ndarray:
    """
    Sub-routine of Lloyd's algorithm that calculates the centers given datapoints and their respective classifications/assignments.
    num_centers is additionally provided for speed-up purposes.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        classifications (np.ndarray): Array of shape (n,) full of integers in range {0, 1, ...,  num_centers - 1}.
            Data point at index i is assigned to classifications[i].
        num_centers (int): Number of centers for reference.
            Might be usefull for pre-allocating numpy array (Faster that appending to list).

    Returns:
        np.ndarray: Array of shape (num_centers, d) containing new centers.
    """
    #raise NotImplementedError("Your Code Goes Here")
    n, d = data.shape
    result = np.zeros((num_centers, d)) 
    clusters = Counter(classifications)
    for i in range(n): 
        result[int(classifications[i])] += data[i]
    for i in range(num_centers):
        result[i] /= clusters[float(i)]
    return result


@problem.tag("hw4-A")
def cluster_data(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Sub-routine of Lloyd's algorithm that clusters datapoints to centers given datapoints and centers.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        centers (np.ndarray): Array of shape (k, d). Each row is a center to which a datapoint can be clustered.

    Returns:
        np.ndarray: Array of integers of shape (n,), with each entry being in range {0, 1, 2, ..., k - 1}.
            Entry j at index i should mean that j^th center is the closest to data[i] datapoint.
    """
    #raise NotImplementedError("Your Code Goes Here")
    n, _ = data.shape 
    result = np.zeros(data.shape[0])
    for i in range(n):
        result[i] = np.argmin(np.linalg.norm(data[i] - centers, axis=1)) 
    return result


# @problem.tag("hw4-A")
def calculate_error(data: np.ndarray, centers: np.ndarray) -> float:
    """Calculates error/objective function on a provided dataset, with trained centers.

    Args:
        data (np.ndarray): Array of shape (n, d). Dataset to evaluate centers on.
        centers (np.ndarray): Array of shape (k, d). Each row is a center to which a datapoint can be clustered.
            These should be trained on training dataset.

    Returns:
        float: Single value representing mean objective function of centers on a provided dataset.
    """
    distances = np.zeros((data.shape[0], centers.shape[0])) 
    for i, c in enumerate(centers):
        distances[:, i] = np.sqrt(np.sum((data - c) **2, axis=1))
    return np.mean(np.min(distances, axis=1))

@problem.tag("hw4-A")
def lloyd_algorithm(
    data: np.ndarray, num_centers: int, epsilon: float = 10e-3, nIter=None
) -> Tuple[np.ndarray, List[float]]:
    """Main part of Lloyd's Algorithm.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        num_centers (int): Number of centers to train/cluster around.
        epsilon (float, optional): Epsilon for stopping condition.
            Training should stop when max(abs(centers - previous_centers)) is smaller or equal to epsilon.
            Defaults to 10e-3.

    Returns:
        np.ndarray: Tuple of 2 numpy arrays:
            Element at index 0: Array of shape (num_centers, d) containing trained centers.
            Element at index 1: List of floats of length # of iterations
                containing errors at the end of each iteration of lloyd's algorithm.

    Note:
        - For initializing centers please use the first `num_centers` data points.
    """
    #raise NotImplementedError("Your Code Goes Here")
    c = data[:num_centers] 
    prev_c = c + np.inf 
    err = []

    while np.linalg.norm(prev_c - c, ord=np.inf ) > epsilon:
        prev_c = np.copy(c)
        clusters = cluster_data(data, c)
        c = calculate_centers(data, clusters, num_centers)
        err.append(calculate_error(data, c))
    return c, err
            



    