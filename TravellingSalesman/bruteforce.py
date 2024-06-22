"""
Class file for brute force algorithm
"""
from tqdm import tqdm


class BF(object):
    """
        Description: BF Class containing relevant data and methods
        Args: list[int], list[list[float]]
    """
    def __init__(self, nodes, distances, start_ind=0, end_ind=0):
        self.nodes = nodes
        self.distances = distances
        self.start_ind = start_ind
        self.end_ind = end_ind
        self.perms = []
        self.perms_length = 0

        self.elite = None
        self.elite_length = 0

    def calc_path_length(self, path):
        """
            Description: Calculates the given path's length
            Args: list[int]
            Returns: float
            Time Complexity: O(n) : n = num nodes
        """
        path_length = 0
        for i in range(len(path)-1):
            path_length += self.distances[path[i]][path[i + 1]]
        return path_length

    def update_elite(self, path, path_length):
        if len(path) < 3:
            return
        if path_length < self.elite_length or self.elite is None:
            self.elite = path
            self.elite_length = path_length

    def generate_perms(self, nodes=None, progress=False):
        """
            Description: Generates all permutations of given nodes
            Args: None
            Returns: list[list[int]]
            Time Complexity: O(n!) : n = num nodes
        """
        result = [[]]
        if nodes is None:
            nodes = self.nodes

        # For tqdm progress bar
        if progress:
            total_permutations = 1
            for i in range(1, len(nodes)+1):
                total_permutations *= i
            progress_bar = tqdm(total=total_permutations, desc="Generating Permutations")

        # Iterate through all nodes
        for node in nodes:
            new_permutations = []
            # Iterate through all permutations of a given node
            for perm in result:
                for i in range(len(perm) + 1):
                    new_permutations.append(perm[:i] + [node] + perm[i:])
                    if progress:
                        progress_bar.update(1)
            result = new_permutations

        self.perms = result
        return self.perms

    def brute_force_search(self, progress=False):
        """
            Description: Performs a brute force search over all possible permutations.
            Args: None
            Returns: list[int], float
            Time Complexity: O(n) : n = length of permutation array
        """
        n = len(self.perms)

        # For tqdm progress bar
        if progress:
            progress_bar = tqdm(total=n, desc="Scanning Permutations")

        # Iterate through all permutations, calculate the length, and update the best path
        for i in range(n):
            perm = self.perms[i]
            perm.insert(0, self.start_ind)
            perm.append(self.end_ind)
            perm_length = self.calc_path_length(perm)
            self.update_elite(perm, perm_length)
            if progress:
                progress_bar.update(1)

        return self.elite, self.elite_length
