"""
File for TSP related methods
"""
import numpy as np
import os


class TSPInstance:
    def __init__(self, name, nodes, coords, dimensions, best_path=None):
        self.name = name
        self.nodes = nodes
        self.coords = coords
        self.dimensions = dimensions
        self.best_path = best_path


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
        name = os.path.basename(tsp_file)
        name = os.path.splitext(name)[0]  # Remove .tsp extension
        nodes, coords, dimensions = self.parse_tsp_file(tsp_file)

        # Parse .opt.tour file if it exists
        best_path = None
        if os.path.exists(opt_tour_file):
            best_path = self.parse_opt_tour_file(opt_tour_file)

        return TSPInstance(name, nodes, coords, dimensions, best_path)

    def parse_tsp_file(self, tsp_file):
        nodes = []
        coords = []

        with open(tsp_file, 'r') as file:
            lines = file.readlines()

        in_node_section = False
        for line in lines:
            line = line.strip()
            if line.startswith("NODE_COORD_SECTION"):
                in_node_section = True
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
                        continue  # Skip lines that are not in expected format

        dimensions = self.calculate_dimensions(coords)
        return nodes, coords, dimensions

    @staticmethod
    def calculate_dimensions(coords):
        min_x = min(coord[0] for coord in coords)
        max_x = max(coord[0] for coord in coords)
        min_y = min(coord[1] for coord in coords)
        max_y = max(coord[1] for coord in coords)
        dimensions = (max_x - min_x, max_y - min_y)
        return dimensions

    @staticmethod
    def parse_opt_tour_file(opt_tour_file):
        with open(opt_tour_file, 'r') as file:
            lines = file.readlines()

        best_path = []
        for line in lines:
            line = line.strip()
            if line.isdigit():
                best_path.append(int(line))

        return best_path

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
