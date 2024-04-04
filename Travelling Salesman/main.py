"""
Main program file for visualizing the Traveling Salesman Problem and experimenting with algorithms like Brute Force,
Divide and Conquer, and Genetic Evolution.
"""
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from os.path import exists
import time as t
import render as r
import bruteforce as bf
import genetic as g

SETTINGS = {
    'Display Dimensions': (750, 600),
    'Max_Framerate': 60,
    'TSP Instance': 'berlin52',
    
    'Random Nodes': False,
    'Dimensions': (1000, 1000),
    'Num Nodes': 100,
    'Start Node': 0,
    'End Node': 0,

    'Population Size': 300,
    'Max Generations': 500,
    'Elite Rate': 0.15,
    'Crossover Rate': 1,
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
    tsp_instance_name = SETTINGS['TSP Instance']
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
    opt_tour = None

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

    # If pre-defined TSP Instance. Must redefine nodes and coords lists.
    else:
        nodes = []
        coords = []
        print(f"TSP Instance: {tsp_instance_name + '.tsp'}")
        with open('TSP Instances/' + tsp_instance_name + '.tsp') as tsp_instance_file:
            lines = [line.rstrip() for line in tsp_instance_file]

        # Parses the .tsp file into nodes list, coords list, and finds the new dimensions
        num_nodes = int(lines[3][11::])
        new_dim = [0, 0]
        for i in range(num_nodes):
            tmp = lines[6+i].split(sep=' ')
            nodes.append(int(tmp[0]))
            tmp_x, tmp_y = int(float(tmp[1])), int(float(tmp[2]))
            coords.append([tmp_x, tmp_y])
            if tmp_x > new_dim[0]:
                new_dim[0] = tmp_x
            if tmp_y > new_dim[1]:
                new_dim[1] = tmp_y
        dimensions = new_dim

        # Load the optimum tour if there is one
        if exists('TSP Instances/' + tsp_instance_name + '.opt.tour'):
            with open('TSP Instances/' + tsp_instance_name + '.opt.tour') as opt_tour_file:
                lines = [line.rstrip() for line in opt_tour_file]

            opt_tour = []
            for i in range(num_nodes):
                tmp = lines[4+i]
                opt_tour.append(int(tmp) - 1)
            opt_tour.append(0)
        print(opt_tour)

    # Calculate the distance between each node
    print(f"Computing distances:")
    distances, elapsed2 = timed(compute_distances, num_nodes, coords)
    print(f"Time elapsed: {elapsed2}s\n")

    # Create Renderer object to display TSP instance and walks
    render_window = r.Renderer(display_dimensions, dimensions, max_framerate, fullscreen=fullscreen,
                               start_ind=start_node, end_ind=end_node)

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

        # Draw the best path so far, the optimum tour if there is one, pass info to renderer as text,
        # and check for pygame events
        text = [f"{genetic_a.best.genes}", f"{genetic_a.best.fitness}", f"{genetic_a.avg_gen_fit}"]
        render_window.draw_frame(coords, genetic_a.best.genes, text, opt_tour)
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

    print(f"Gen 0 Best: {genetic_a.generations[0][0].fitness}")
    print(f"Gen 0 Worst: {genetic_a.generations[0][-1].fitness}")
    print()
    print(f"Gen {max_gens} Best: {genetic_a.best.fitness}")
    print(f"Gen {max_gens} Worst: {genetic_a.generations[-1][-1].fitness}")
    print()

    if opt_tour:
        opt_tour_ind = g.Individual(opt_tour, np.inf)
        genetic_a.evaluate_individual(opt_tour_ind)
        print(f"Optimum Tour Fitness: {opt_tour_ind.fitness}")

    # Listen for user input to quit program
    while True:
        r.event_listen()


if __name__ == '__main__':
    main()
