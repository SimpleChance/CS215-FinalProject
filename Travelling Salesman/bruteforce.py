"""
Class file for Brute Force solution to the Travelling Salesman problem
"""
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

    def generate_perms(self):
        tmp = it.permutations(self.nodes)
        for perm in tmp:
            p = list(perm)
            p.insert(0, self.start_ind)
            p.append(self.end_ind)
            self.perms.append(p)
        self.perms_length = len(self.perms)

    def calc_path_length(self, path):
        path_length = 0
        for i in range(len(path)-1):
            path_length += self.distances[path[i]][path[i + 1]]
        return path_length

    def brute_force_search(self):
        for perm in self.perms:
            path_length = self.calc_path_length(perm)
            if path_length < self.elite_length or self.elite is None:
                self.elite = perm
                self.elite_length = path_length
        return self.elite, self.elite_length

    def brute_force_search_step(self, step):
        if step > self.perms_length:
            return

        path_length = self.calc_path_length(self.perms[step])
        if path_length < self.elite_length or self.elite is None:
            self.elite = self.perms[step]
            self.elite_length = path_length
