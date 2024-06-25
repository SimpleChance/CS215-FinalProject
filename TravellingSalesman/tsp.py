"""
File for TSP related methods
"""
import numpy as np
import os


class TSPInstance:
    def __init__(self, name, num_nodes, nodes, coords, distance_matrix, space_dimensions, best_path=None,
                 best_path_length=None):
        self.name = name
        self.num_nodes = num_nodes
        self.nodes = nodes
        self.coords = coords
        self.distance_matrix = distance_matrix
        self.space_dimensions = space_dimensions
        self.best_path = best_path
        self.best_path_length = best_path_length


def calculate_path_length(path, distance_matrix):
    distance = 0
    for i in range(len(path) - 1):
        distance += distance_matrix[path[i] - 1][path[i + 1] - 1]
    return distance


class TSPParser:
    def __init__(self):
        self.directory = os.path.join(os.path.dirname(__file__), 'TSP Instances')
        self.instances = []
        self.parse_instances()

    def parse_instances(self):
        for filename in os.listdir(self.directory):
            if filename.endswith(".tsp"):
                tsp_file = os.path.join(self.directory, filename)
                opt_tour_file = os.path.join(self.directory, filename.replace(".tsp", ".opt.tour"))
                instance_data = self.parse_files(tsp_file, opt_tour_file)
                self.instances.append(instance_data)

    def parse_files(self, tsp_file, opt_tour_file):
        # Parse .tsp file
        name = os.path.basename(tsp_file).replace('.tsp', '')
        nodes, coords = self.parse_tsp_file(tsp_file)
        num_nodes = len(nodes)
        distance_matrix = self.calculate_distance_matrix(coords)
        space_dimensions = self.calculate_space_dimensions(coords)

        # Parse .opt.tour file if it exists
        best_path = None
        best_path_length = None
        if os.path.exists(opt_tour_file):
            best_path = self.parse_opt_tour_file(opt_tour_file)
            best_path_length = calculate_path_length(best_path, distance_matrix)

        return TSPInstance(name, num_nodes, nodes, coords, distance_matrix, space_dimensions, best_path,
                           best_path_length)

    @staticmethod
    def parse_tsp_file(tsp_file):
        nodes = []
        coords = []

        with open(tsp_file, 'r') as file:
            lines = file.readlines()

        in_node_section = False
        for line in lines:
            line = line.strip()
            if line.startswith("NODE_COORD_SECTION"):
                in_node_section = True
                continue
            elif line.startswith("EOF"):
                break

            if in_node_section:
                parts = line.split()
                if len(parts) > 1:
                    try:
                        node_id = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        nodes.append(node_id)
                        coords.append((x, y))
                    except ValueError:
                        continue

        return nodes, coords

    @staticmethod
    def parse_opt_tour_file(opt_tour_file):
        with open(opt_tour_file, 'r') as file:
            lines = file.readlines()

        best_path = []
        in_tour_section = False
        for line in lines:
            line = line.strip()
            if line.startswith("TOUR_SECTION"):
                in_tour_section = True
                continue
            if line == "-1" or line == "EOF":
                break

            if in_tour_section:
                try:
                    best_path.append(int(line))
                except ValueError:
                    continue

        return best_path

    def calculate_distance_matrix(self, coords):
        num_nodes = len(coords)
        dist_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    dist_matrix[i][j] = self.distance(coords[i], coords[j])
        return dist_matrix

    @staticmethod
    def calculate_space_dimensions(coords):
        x_coords, y_coords = zip(*coords)
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        return (max_x - min_x, max_y - min_y)

    @staticmethod
    def distance(p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def get_instances(self):
        return self.instances


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
