"""
Script for testing the Genetic Algorithm with variable number of nodes
"""
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import tsp
import genetic as g


SETTINGS = {
    'Batch Size': 100,

    'Max Num Nodes': 250,
    'Dimensions': (500, 500),
    'Start Node': 0,
    'End Node': 0,

    'Population': 25,
    'Max Generations': 25,
    'Elite Rate': 0,
    'Crossover Rate': 1,
    'Mutation Rate': 1
}


def main():
    batch_size = SETTINGS['Batch Size']
    max_num_nodes = SETTINGS['Max Num Nodes']
    dimensions = SETTINGS['Dimensions']
    start_node, end_node = SETTINGS['Start Node'], SETTINGS['End Node']
    population = SETTINGS['Population']
    max_gens = SETTINGS['Max Generations']
    elite_rate = SETTINGS['Elite Rate']
    cross_rate = SETTINGS['Crossover Rate']
    mut_rate = SETTINGS['Mutation Rate']

    data_x = [_ for _ in range(5, max_num_nodes+1)]
    data_y = []
    for _ in range(batch_size):
        data_y.append([])

    progress_bar1 = tqdm(total=batch_size, desc='Running Batch Test...')
    for i in range(batch_size):
        for num_nodes in range(5, max_num_nodes+1):
            coords = tsp.generate_random_coords(dimensions, num_nodes)
            distance_matrix = tsp.compute_distances(num_nodes, coords)

            genetic_obj = g.GA(num_nodes, population, cross_rate, mut_rate, distance_matrix, elite_rate, max_gens,
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

    avg_elapsed = []
    for i in range(len(data_x)):
        tmp = 0
        for j in range(batch_size):
            tmp += data_y[j][i]
        tmp /= batch_size
        avg_elapsed.append(tmp)
    x = np.array(data_x)
    y = np.array(avg_elapsed)
    plt.scatter(x, y, label='Average time elapsed: linear (n) fit', color=(0, 0, 0))

    plt.title(f"Time to Perform Genetic Algorithm on N Nodes | Pop: {population} | Gens: {max_gens} |\n| "
              f"Elite Rate: {elite_rate} | Cross Rate: {cross_rate} | Mut Rate: {mut_rate} | Batch Size: {batch_size}")
    plt.xlabel("Num Nodes")
    plt.ylabel("Time (s)")
    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    main()
