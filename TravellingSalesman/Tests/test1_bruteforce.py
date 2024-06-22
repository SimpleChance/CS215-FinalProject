"""
Script for testing the Brute Force algorithm to solve TSP
"""
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import time
from TravellingSalesman import tsp
from TravellingSalesman import bruteforce as bf


SETTINGS = {
    'Batch Size': 100,
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

    data_x = [_ for _ in range(4, max_num_nodes+1)]
    data_y = []
    for i in range(batch_size):
        data_y.append([])

    progress_bar = tqdm(total=batch_size, desc='Running Tests...')
    for i in range(batch_size):
        for num_nodes in range(4, max_num_nodes+1):
            t1 = time.perf_counter()

            nodes = [i for i in range(num_nodes)]
            nodes.remove(start_node)
            if start_node != end_node:
                nodes.remove(end_node)

            coords = tsp.generate_random_coords(dimensions, num_nodes)
            distance_matrix = tsp.compute_distances(num_nodes, coords)

            brute_f = bf.BF(nodes, distance_matrix, start_node, end_node)

            brute_f.generate_perms()
            brute_f.brute_force_search()

            t2 = time.perf_counter()
            elapsed = t2 - t1

            data_y[i].append(elapsed)

        progress_bar.update(1)

    # Find average elapsed for all batches
    avg_elapsed = []
    for i in range(len(data_x)):
        tmp = 0
        for j in range(batch_size):
            tmp += data_y[j][i]
        tmp /= batch_size
        avg_elapsed.append(tmp)
    x = np.array(data_x)
    y = np.array(avg_elapsed)

    # Output data to .txt file
    with open('../Test Results/bf_data.txt', 'w') as f:
        f.write('Brute Force test with variable # of nodes, N\n')
        f.write(f'Batch size: {batch_size}\n')
        f.write('\n')
        for i in range(batch_size):
            f.write(f'\nBatch {i}: \n')
            for j in range(len(x)):
                f.write(f'# nodes: {x[j]}, Time elapsed: {data_y[i][j]} s \n')
        f.write(f'\nAverage time elapsed between all batches for # nodes, N: \n')
        for i in range(len(x)):
            f.write(f'# nodes: {x[i]}, Average time elapsed: {y[i]} s\n')

    # Plot data
    for i in range(batch_size):
        plt.plot(x, data_y[i])
    plt.scatter(x, y, color=(0, 0, 0), label='Average time elapsed')
    plt.title(f"Time to Perform Brute Force Search on N Nodes | Batch Size = {batch_size}")
    plt.xlabel("Num Nodes")
    plt.ylabel("Time (s)")
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()
