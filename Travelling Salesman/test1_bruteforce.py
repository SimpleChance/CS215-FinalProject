"""
Script for testing the Brute Force algorithm to solve TSP
"""
from matplotlib import pyplot as plt
import time
import tsp
import bruteforce as bf


SETTINGS = {
    'Batch Size': 10,
    'Random Nodes': True,
    'Max Num Nodes': 11,
    'Dimensions': (500, 500),
    'Start Node': 0,
    'End Node': 0
}


def main():
    batch_size = SETTINGS['Batch Size']
    max_num_nodes = SETTINGS['Max Num Nodes']
    dimensions = SETTINGS['Dimensions']
    start_node, end_node = SETTINGS['Start Node'], SETTINGS['End Node']

    data_x = []
    data_y = []
    for i in range(batch_size):
        data_x.append([_ for _ in range(3, max_num_nodes+1)])
        data_y.append([])
    for i in range(batch_size):
        for num_nodes in range(3, max_num_nodes+1):
            t1 = time.perf_counter()

            nodes = [i for i in range(num_nodes)]
            nodes.remove(start_node)
            if start_node != end_node:
                nodes.remove(end_node)

            coords = tsp.generate_random_coords(dimensions, num_nodes)
            distance_matrix = tsp.compute_distances(num_nodes, coords)

            brute_f = bf.BF(nodes, distance_matrix, start_node, end_node)

            perms = brute_f.generate_perms()
            best, best_length = brute_f.brute_force_search()

            t2 = time.perf_counter()
            elapsed = t2 - t1

            data_y[i].append(elapsed)

            print(f"Best: {best}\nLength: {best_length}")
            print(f"Number of perms: {len(perms)}")
            print(f"Elapsed time: {elapsed}")

    for i in range(batch_size):
        plt.plot(data_x[i], data_y[i])
    plt.title(f"Time to Perform Brute Force Search on N Nodes | Batch Size = {batch_size}")
    plt.xlabel("Num Nodes")
    plt.ylabel("Time (s)")
    plt.show()


if __name__ == "__main__":
    main()
