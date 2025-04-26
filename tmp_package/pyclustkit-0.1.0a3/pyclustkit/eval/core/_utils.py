import numpy as np


def upper_triu(x):
    m = x.shape[0]
    r, c = np.triu_indices(m, 1)
    return x[r, c]


def sum_of_upper_triu(x):
    m = x.shape[0]
    r, c = np.triu_indices(m, 1)
    return np.sum(x[r,c])


def find_midpoint(point1, point2):
    """
    Calculate the midpoint of the line formed by two points in n-dimensional space.

    Parameters:
    point1 (tuple): A tuple representing the coordinates of the first point.
    point2 (tuple): A tuple representing the coordinates of the second point.

    Returns:
    tuple: A tuple representing the midpoint.
    """
    if len(point1) != len(point2):
        raise ValueError("Both points must have the same dimension.")

    midpoint = tuple((p1 + p2) / 2 for p1, p2 in zip(point1, point2))

    return midpoint