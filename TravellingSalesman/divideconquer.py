"""
Class file for divide and conquer algorithm
"""
import numpy as np


class DC(object):
    def __init__(self, coords, distance_matrix):
        self.coords = coords
        self.distance_matrix = distance_matrix

    @staticmethod
    def partition_nodes(nodes: list[tuple[int, int]]):
        nodes_sorted_by_x = sorted(nodes, key=lambda x: x[0])
        nodes_sorted_by_y = sorted(nodes, key=lambda x: x[1])
        mid = len(nodes_sorted_by_x) // 2
        if (abs(nodes_sorted_by_x[0][0] - nodes_sorted_by_x[-1][0]) >
                abs(nodes_sorted_by_y[0][1] - nodes_sorted_by_y[-1][1])):
            return nodes_sorted_by_x[:mid], nodes_sorted_by_x[mid:]
        else:
            return nodes_sorted_by_y[:mid], nodes_sorted_by_y[mid:]

    def merge_tours(self, tour1, tour2):
        if len(tour1) == 1 and len(tour2) == 1:
            # If both tours have only one node each
            dist = self.distance_matrix[tour1[0]][tour2[0]]
            return [tour1[0], tour2[0]] if dist <= 0 else [tour2[0], tour1[0]]

        min_distance = np.inf
        closest_nodes = None
        # Iterate over pairs of nodes from both tours
        for node1 in tour1:
            for node2 in tour2:
                dist = self.distance_matrix[node1][node2]
                if dist < min_distance:
                    min_distance = dist
                    closest_nodes = (node1, node2)

        if closest_nodes[0] == tour1[0]:
            reversed(tour1)
        if closest_nodes[1] == tour2[-1]:
            reversed(tour2)

        return tour1 + tour2

    def divide_and_conquer(self, nodes: list[tuple[int, int]]):
        if len(nodes) < 1:
            raise Exception('Recurse on node list of length 0')
        elif len(nodes) == 1:
            return [self.coords.index(nodes[0])]
        elif len(nodes) == 2:
            return [self.coords.index(nodes[0]), self.coords.index(nodes[1])]
        else:
            partition1, partition2 = self.partition_nodes(nodes)
            tour1 = self.divide_and_conquer(partition1)
            tour2 = self.divide_and_conquer(partition2)
            combined_tour = self.merge_tours(tour1, tour2)

            return combined_tour

    def run(self):
        solution = self.divide_and_conquer(self.coords)
        solution.append(solution[0])
        return solution
