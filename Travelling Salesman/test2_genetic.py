"""
Script for testing the Genetic Algorithm to solve TSP
"""
from matplotlib import pyplot as plt
import time
import tsp
import genetic as g


SETTINGS = {
    'Batch Size': 10,

    'Max Num Nodes': 25,
    'Dimensions': (500, 500),
    'Start Node': 0,
    'End Node': 0,

    'Population': 25,
    'Max Generations': 50,
    'Elite Rate': 0.15,
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

    data_x = []
    data_y = []
    for _ in range(batch_size):
        data_x.append([_ for _ in range(5, max_num_nodes+1)])
        data_y.append([])

    for i in range(batch_size):
        for num_nodes in range(5, max_num_nodes+1):
            t1 = time.perf_counter()

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

    for i in range(batch_size):
        plt.plot(data_x[i], data_y[i])
    plt.title(f"Time to Perform Genetic Algorithm on N Nodes | Pop: {population} | Gens: {max_gens} | "
              f"Elite Rate: {elite_rate} | Cross Rate: {cross_rate} | Mut Rate: {mut_rate}")
    plt.xlabel("Num Nodes")
    plt.ylabel("Time (s)")
    plt.show()


if __name__ == "__main__":
    main()
