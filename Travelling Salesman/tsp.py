"""
File for TSP related methods
"""
import numpy as np


def parse_instance(text):
    """
        Description: Parses a .tsp instance for relevant information.
        Args: list[string]
        Returns: int, list[int], list[list[int]], list[int]
    """
    nodes = []
    coords = []
    dimensions = [0, 0]

    num_nodes = int(text[3][11::])
    for i in range(num_nodes):
        tmp = text[6+i].split(sep=' ')
        nodes.append(int(tmp[0]))
        tmp_x, tmp_y = int(float(tmp[1])), int(float(tmp[2]))
        coords.append([tmp_x, tmp_y])
        if tmp_x > dimensions[0]:
            dimensions[0] = tmp_x
        if tmp_y > dimensions[1]:
            dimensions[1] = tmp_y

    return num_nodes, nodes, coords, dimensions


def parse_opt_tour(text):
    """
        Description: Parses a .opt.tsp instance into a list of nodes.
        Args: list[string]
        Returns: list[int]
    """
    opt_tour = []
    num_nodes = int(text[2].split(sep=' ')[2])
    for i in range(num_nodes):
        tmp = text[4 + i]
        opt_tour.append(int(tmp) - 1)
    opt_tour.append(0)
    return opt_tour


def generate_random_coords(dimensions, num_nodes):
    """
        Description: Generates random coordinates within the specified dimensions for a number of specified nodes
        Args: list[int], int
        Returns: list[list[int]]
        Time Complexity: O(n) : n = num nodes
    """
    return [[np.random.randint(0, dimensions[0]), np.random.randint(0, dimensions[1])]
            for _ in range(num_nodes)]


def compute_distances(num_nodes, coords):
    """
        Description: Computes the distances between all given nodes and stores them in a matrix.
        Args: int, list[list[int]]
        Returns: list[list[float]]
        Time Complexity: O(n^2) : n = num nodes
    """
    # Array of size n by n where array[i][j] is the distance between node i and j
    distance_matrix = np.empty((num_nodes, num_nodes), dtype=float)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                d = (x2 - x1)**2 + (y2 - y1)**2
                distance_matrix[i][j] = np.sqrt(d)
    return distance_matrix
