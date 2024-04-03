"""
Main program file for visualizing the Traveling Salesman Problem and experimenting with algorithms like Brute Force,
Divide and Conquer, and Genetic Evolution.
"""
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import time as t
import render as r
import bruteforce as bf
import genetic as g

SETTINGS = {
    'Display Dimensions': (500, 500),
    'Max_Framerate': 60,
    'TSP Instance': 'berlin52.tsp',
    
    'Random Nodes': True,
    'Dimensions': (1000, 1000),
    'Num Nodes': 250,
    'Start Node': 0,
    'End Node': 0,

    'Population Size': 250,
    'Max Generations': 1000,
    'Elite Rate': 1,
    'Crossover Rate': 0.5,
    'Mutation Rate': 1,

    'Fullscreen': False
}


# Used to time other functions
def timed(func, *args, **kwargs):
    t1 = t.perf_counter()
    result = func(*args, **kwargs)
    t2 = t.perf_counter()
    elapsed = t2 - t1
    return result, elapsed


# Generates random coordinates for every node. Returns a list of coordinate pairs
def generate_random_coords(dimensions, num_nodes):
    return [[np.random.randint(0, dimensions[0]), np.random.randint(0, dimensions[1])] for _ in range(num_nodes)]


# Computes the distance matrix for all nodes. Returns 2-D array of distances where each index is a node
def compute_distances(num_nodes, coords):
    distance_matrix = np.empty((num_nodes, num_nodes), dtype=float)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                d = (x2 - x1)**2 + (y2 - y1)**2
                distance_matrix[i][j] = np.sqrt(d)
    return distance_matrix


def main():
    display_dimensions = SETTINGS['Display Dimensions']
    max_framerate = SETTINGS['Max_Framerate']
    tsp_instance_path = SETTINGS['TSP Instance']
    random_nodes = SETTINGS['Random Nodes']
    dimensions = SETTINGS['Dimensions']
    num_nodes = SETTINGS['Num Nodes']
    pop_size = SETTINGS['Population Size']
    max_gens = SETTINGS['Max Generations']
    elite_rate = SETTINGS['Elite Rate']
    cross_rate = SETTINGS['Crossover Rate']
    mutate_rate = SETTINGS['Mutation Rate']
    start_node, end_node = SETTINGS['Start Node'], SETTINGS['End Node']
    fullscreen = SETTINGS['Fullscreen']

    render_window = r.Renderer(display_dimensions, dimensions, max_framerate, fullscreen=fullscreen,
                               start_ind=start_node, end_ind=end_node)

    # If random node set
    if random_nodes:
        print(f"\nNumber of nodes: {num_nodes}\n")

        # Generate set of nodes with random coords within specified dimensions
        print(f"Generating random coordinates:")
        coords, elapsed1 = timed(generate_random_coords, dimensions, num_nodes)
        print(f"Time elapsed: {elapsed1}s\n")

        # Create a set of available nodes to permute (start and ends node are not included)
        # Nodes are given a value between 0 and num_nodes, this is their unique identifier
        nodes = [i for i in range(num_nodes)]
        nodes.remove(start_node)
        if start_node != end_node:
            nodes.remove(end_node)

    # If pre-defined TSP Instance
    # Currently, the node indexes and coordinates are parsed into 2 lists
    else:
        nodes = []
        coords = []
        print(f"TSP Instance: {tsp_instance_path}")
        with open('TSP Instances/'+tsp_instance_path) as tsp_instance_file:
            lines = [line.rstrip() for line in tsp_instance_file]

        num_nodes = int(lines[3][11::])
        for i in range(num_nodes):
            tmp = lines[6+i].split(sep=' ')
            nodes.append(int(tmp[0]))
            coords.append([float(tmp[1]), float(tmp[2])])
        print(num_nodes)
        print(nodes)
        print(coords)

    # Calculate the distance between each node
    print(f"Computing distances:")
    distances, elapsed2 = timed(compute_distances, num_nodes, coords)
    print(f"Time elapsed: {elapsed2}s\n")

    '''
    # Create a BF object with the available nodes and distances
    brute_f = bf.BF(nodes, distances, start_ind=start_node, end_ind=end_node)

    # Exhaustively generate all permutations of the available nodes
    print(f"Generating permutations:")
    result, elapsed3 = timed(brute_f.generate_perms)
    print(f"Time elapsed: {elapsed3}s")
    print(f"Length of permutation array: {brute_f.perms_length}\n")

    # Iterate over all permutations and return the best one
    print(f"Scanning permutations:")
    result, elapsed4 = timed(brute_f.brute_force_search)
    print(f"Time elapsed: {elapsed4}s\n")

    print(f"Best path: {result[0]}\nLength: {result[1]}")
    '''

    # Create a GA object with specified params
    genetic_a = g.GA(num_nodes, pop_size, cross_rate, mutate_rate, distances, elite_rate, max_gens,
                     start_ind=start_node, end_ind=end_node)

    # Initialize Generation 0
    print(f"Initializing Generation 0:")
    result, elapsed5 = timed(genetic_a.initialize_population)
    print(f"Time elapsed: {elapsed5}s\n")

    # Evaluate the population
    genetic_a.evaluate_population()

    # Sort population
    genetic_a.sort_population_by_fitness()

    # Find average fitness and best_fit
    genetic_a.find_avg_fit()
    genetic_a.find_gen_best_fit()

    # Use tqdm to track overall completion
    for _ in tqdm(range(max_gens)):

        # Calculate Parent probabilities
        genetic_a.find_parent_probabilities()

        # Select parents and perform crossover
        offspring = genetic_a.generate_offspring()

        # Mutate the offspring
        offspring = genetic_a.mutate_batch(offspring)

        # Evaluate offspring
        offspring = genetic_a.evaluate_population(population=offspring)

        # Sort the offspring by fitness
        offspring = genetic_a.sort_population_by_fitness(offspring)

        # Generate new population
        genetic_a.select_new_population(offspring)

        # Sort population once again for rendering the best path
        genetic_a.sort_population_by_fitness()

        # Draw the best path so far and check for pygame events
        render_window.draw_frame(coords, genetic_a.best.genes)
        r.event_listen()

    # Compile fitness data into matplot chart
    x = list(range(max_gens+1))
    y1 = genetic_a.avg_gen_fits
    y2 = genetic_a.gen_best_fits
    plt.xlabel("Generation n")
    plt.ylabel("Average Fitness")
    plt.title(f"Nodes: {num_nodes} | Pop: {pop_size} | Elite: {elite_rate} | Cross: {cross_rate} | Mut: {mutate_rate}")
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.show()

    print(f"Gen 0 Best: {genetic_a.generations[0][0]}")
    print(f"Gen 0 Worst: {genetic_a.generations[0][-1]}")
    print()
    print(f"Gen {max_gens} Best: {genetic_a.best}")
    print(f"Gen {max_gens} Worst: {genetic_a.generations[-1][-1]}")

    # Listen for user input to quit program
    while True:
        r.event_listen()


if __name__ == '__main__':
    main()
