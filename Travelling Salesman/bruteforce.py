"""
Class file for Brute Force solution to the Travelling Salesman problem
"""
from tqdm import tqdm
import itertools as it


class BF(object):
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
        path_length = 0
        for i in range(len(path)-1):
            path_length += self.distances[path[i]][path[i + 1]]
        return path_length

    def generate_perms(self, nodes=None, progress=False):
        result = [[]]
        if nodes is None:
            nodes = self.nodes

        if progress:
            total_permutations = 1
            for i in range(1, len(nodes)+1):
                total_permutations *= i
            progress_bar = tqdm(total=total_permutations, desc="Generating Permutations")

        for node in nodes:
            new_permutations = []
            for perm in result:
                for i in range(len(perm) + 1):
                    new_permutations.append(perm[:i] + [node] + perm[i:])
                    if progress:
                        progress_bar.update(1)
            result = new_permutations

        self.perms = result
        return self.perms

    def brute_force_search(self, progress=False):
        n = len(self.perms)

        if progress:
            progress_bar = tqdm(total=n, desc="Scanning Permutations")

        for i in range(n):
            perm = self.brute_force_search_step(i)
            path_length = self.calc_path_length(perm)
            if path_length < self.elite_length or self.elite is None:
                self.elite = perm
                self.elite_length = path_length
            if progress:
                progress_bar.update(1)

        return self.elite, self.elite_length

    def brute_force_search_step(self, step):
        perm = self.perms[step]
        perm.insert(0, self.start_ind)
        perm.append(self.end_ind)
        return perm
