"""
File for TSP related methods
"""

from os.path import exists
import numpy as np


def parse_instance(text):
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
    opt_tour = []
    num_nodes = int(text[2].split(sep=' ')[2])
    for i in range(num_nodes):
        tmp = text[4 + i]
        opt_tour.append(int(tmp) - 1)
    opt_tour.append(0)
    return opt_tour


def generate_random_coords(dimensions, num_nodes):
    return [[np.random.randint(0, dimensions[0]), np.random.randint(0, dimensions[1])]
            for _ in range(num_nodes)]


def compute_distances(num_nodes, coords):
    distance_matrix = np.empty((num_nodes, num_nodes), dtype=float)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                d = (x2 - x1)**2 + (y2 - y1)**2
                distance_matrix[i][j] = np.sqrt(d)
    return distance_matrix
