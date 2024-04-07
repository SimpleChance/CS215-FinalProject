"""
Script to test Genetic Algorithm with variable population
"""
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import tsp
import genetic as g


SETTINGS = {
    'Batch Size': 50,

    'Num Nodes': 25,
    'Dimensions': (500, 500),
    'Start Node': 0,
    'End Node': 0,

    'Max Population': 100,
    'Max Generations': 10,
    'Elite Rate': 0,
    'Crossover Rate': 1,
    'Mutation Rate': 1
}


def main():
    batch_size = SETTINGS['Batch Size']
    num_nodes = SETTINGS['Num Nodes']
    dimensions = SETTINGS['Dimensions']
    start_node, end_node = SETTINGS['Start Node'], SETTINGS['End Node']
    max_population = SETTINGS['Max Population']
    max_gens = SETTINGS['Max Generations']
    elite_rate = SETTINGS['Elite Rate']
    cross_rate = SETTINGS['Crossover Rate']
    mut_rate = SETTINGS['Mutation Rate']

    data_x = [_ for _ in range(4, max_population)]
    data_y = []
    for i in range(batch_size):
        data_y.append([])

    progress_bar1 = tqdm(total=batch_size, desc='Running Batch Test...')
    for i in range(batch_size):
        for j in range(4, max_population):
            coords = tsp.generate_random_coords(dimensions, num_nodes)
            distance_matrix = tsp.compute_distances(num_nodes, coords)
            genetic_obj = g.GA(num_nodes, j, cross_rate, mut_rate, distance_matrix, elite_rate, max_gens,
                               start_ind=start_node, end_ind=end_node)

            genetic_obj.initialize_population()

            # Initialize Generation 0
            genetic_obj.initialize_population()

            # Evaluate the population
            genetic_obj.evaluate_population()

            # Sort population
            genetic_obj.sort_population_by_fitness()

            # Find average fitness and best_fit
            genetic_obj.find_avg_fit()
            genetic_obj.find_gen_best_fit()

            t1 = time.perf_counter()
            for _ in range(max_gens):

                # Calculate Parent probabilities
                genetic_obj.find_parent_probabilities()

                # Select parents and perform crossover
                offspring = genetic_obj.generate_offspring()

                # Mutate the offspring
                offspring = genetic_obj.mutate_batch(offspring)

                # Evaluate offspring
                offspring = genetic_obj.evaluate_population(population=offspring)

                # Sort the offspring by fitness
                offspring = genetic_obj.sort_population_by_fitness(offspring)

                # Generate new population
                genetic_obj.select_new_population(offspring)

                # Sort population once again for rendering the best path
                genetic_obj.sort_population_by_fitness()
            t2 = time.perf_counter()
            elapsed = t2 - t1
            data_y[i].append(elapsed)
        progress_bar1.update(1)

    for i in range(batch_size):
        plt.plot(data_x, data_y[i])

    plt.title(f"Time to Perform Genetic Algorithm with Population N | Num Nodes: {num_nodes} | Gens: {max_gens} | "
              f"Elite Rate: {elite_rate} | Cross Rate: {cross_rate} | Mut Rate: {mut_rate}")
    plt.xlabel("Population Size n")
    plt.ylabel("Time (s)")
    plt.show()

    avg_elapsed = []
    for i in range(len(data_x)):
        tmp = 0
        for j in range(batch_size):
            tmp += data_y[j][i]
        tmp /= batch_size
        avg_elapsed.append(tmp)
    x = np.array(data_x)
    y = np.array(avg_elapsed)
    a, b = np.polyfit(x*np.log(x), y, 1)

    plt.scatter(x, y, label='Average time elapsed: n*log(n) fit')
    plt.title(f"Average Time to Perform Genetic Algorithm with Population N | Batch Size: {batch_size}")
    plt.xlabel("Population Size")
    plt.ylabel("Time (s)")
    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    main()
