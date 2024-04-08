"""
Main program file for visualizing the Traveling Salesman Problem and experimenting with algorithms like Brute Force,
Divide and Conquer, and Genetic Evolution.
"""
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from os.path import exists
import time as t
import tsp
import render as r
import bruteforce as bf
import genetic as g

SETTINGS = {
    'Display Dimensions': (750, 600),   # Display Resolution. Keep default for small window.
    'Max_Framerate': 30,                # Maximum framerate the animation will play at.
    'TSP Instance': 'att48',            # .tsp file name
    "Display": True,                    # Boolean to determine if the animation should play.
    
    'Random Nodes': False,              # Determines if random nodes should be used. Keep true if no .tsp file is given.
    'Dimensions': (1000, 1000),         # Max and min coords for nodes.
    'Num Nodes': 50,                    # Brute force will be disabled for node spaces larger than 12.
    'Start Node': 0,                    # Index of start node.
    'End Node': 0,                      # Index of end node.

    'Population Size': 150,             # Population size for the genetic algorithm.
    'Max Generations': 1000,            # Maximum number of generations for the genetic algorithm.
    'Elite Rate': 0,                    # Determines how many individuals from the previous epoch will survive. (0-1)
    'Crossover Rate': 1,                # Determines how many individuals will reproduce. (0-1)
    'Mutation Rate': 1,                 # Determines the frequency of offspring mutations. (0-1)

    'Fullscreen': False                 # Determines if fullscreen should be toggled.
                                        # (May alter position of text during animation)
}


# Used to time other functions
def timed(func, *args, **kwargs):
    t1 = t.perf_counter()
    result = func(*args, **kwargs)
    t2 = t.perf_counter()
    elapsed = t2 - t1
    return result, elapsed


def main():
    """
    Start of precompute and initialization
    """
    # Checks for valid program settings
    display_dimensions = SETTINGS['Display Dimensions']
    max_framerate = SETTINGS['Max_Framerate']
    if max_framerate < 1:
        print(f"Invalid maximum framerate given. Setting max_framerate to 30...")
        max_framerate = 30

    tsp_instance_name = SETTINGS['TSP Instance']
    random_nodes = SETTINGS['Random Nodes']
    dimensions = SETTINGS['Dimensions']

    start_node, end_node = SETTINGS['Start Node'], SETTINGS['End Node']
    num_nodes = SETTINGS['Num Nodes']
    if num_nodes <= 3:
        if start_node == end_node:
            print(f"Invalid number of nodes given. Setting num_nodes to 4...")
            num_nodes = 4
        else:
            print(f"Invalid number of nodes given. Setting num_nodes to 5...")
            num_nodes = 5
    if start_node < 0 or start_node > num_nodes-1:
        print(f"Invalid start node given. Setting start_node to 0...")
        start_node = 0
    if end_node < 0 or end_node > num_nodes-1:
        print(f"Invalid end node given. Setting end_node to 0...")
        end_node = 0

    pop_size = SETTINGS['Population Size']
    if pop_size <= 1:
        print(f"Invalid population size given. Setting pop_size to 2...")
        pop_size = 2
    max_gens = SETTINGS['Max Generations']
    elite_rate = SETTINGS['Elite Rate']
    cross_rate = SETTINGS['Crossover Rate']
    mutate_rate = SETTINGS['Mutation Rate']
    fullscreen = SETTINGS['Fullscreen']
    opt_tour = None

    # If random node set
    if random_nodes:
        print(f"\nNumber of nodes: {num_nodes}\n")

        # Generate set of nodes with random coords within specified dimensions
        print(f"Generating random coordinates:")
        coords, elapsed1 = timed(tsp.generate_random_coords, dimensions, num_nodes)
        print(f"Time elapsed: {elapsed1}s\n")

        # Create a set of available nodes to permute (start and ends node are not included)
        # Nodes are given a value between 0 and num_nodes, this is their unique identifier
        nodes = [i for i in range(num_nodes)]
        nodes.remove(start_node)
        if start_node != end_node:
            nodes.remove(end_node)

    # If pre-defined TSP Instance. Must redefine nodes and coords lists.
    else:
        if exists('TSP Instances/' + tsp_instance_name + '.tsp'):
            print(f"TSP Instance: {tsp_instance_name + '.tsp'}")
            with open('TSP Instances/' + tsp_instance_name + '.tsp') as tsp_instance_file:
                lines = [line.rstrip() for line in tsp_instance_file]

            num_nodes, nodes, coords, dimensions = tsp.parse_instance(lines)
        else:
            print(f"File {tsp_instance_name} not found. Exiting...")
            exit()

        # Load the optimum tour if there is one
        if exists('TSP Instances/' + tsp_instance_name + '.opt.tour'):
            with open('TSP Instances/' + tsp_instance_name + '.opt.tour') as opt_tour_file:
                lines = [line.rstrip() for line in opt_tour_file]
        else:
            print(f"No optimum path given.")

        if lines:
            opt_tour = tsp.parse_opt_tour(lines)

    # Calculate the distance between each node
    print(f"Computing distances:")
    distances, elapsed2 = timed(tsp.compute_distances, num_nodes, coords)
    print(f"Time elapsed: {elapsed2}s\n")

    """
    Start of Brute Force algorithm
    """
    # Limit brute force to max of 12 nodes
    if num_nodes <= 12:
        # Create a BF object with the available nodes and distances
        brute_f = bf.BF(nodes, distances, start_ind=start_node, end_ind=end_node)

        # Exhaustively generate all permutations of the available nodes
        brute_f.generate_perms(progress=True)

        # Iterate over all permutations and return the best one
        brute_f.brute_force_search(progress=True)

        if opt_tour is None:
            opt_tour = brute_f.elite

        print(f"Best path found with brute force: {brute_f.elite}\nLength: {brute_f.elite_length}")
        print()

    """
    Start of Genetic Algorithm
    """
    # Create a GA object with specified params
    genetic_a = g.GA(num_nodes, pop_size, cross_rate, mutate_rate, distances, elite_rate, max_gens,
                     start_ind=start_node, end_ind=end_node)

    if opt_tour:
        opt_tour_ind = g.Individual(opt_tour, np.inf)
        genetic_a.evaluate_individual(opt_tour_ind, opt_tour=True)

    # Create Renderer object to display TSP instance and walks
    if SETTINGS['Display']:
        render_window = r.Renderer(display_dimensions, dimensions, coords, max_framerate, fullscreen=fullscreen,
                                   start_ind=start_node, end_ind=end_node)

    print(f"Starting Genetic Algorithm:")
    # Initialize Generation 0
    genetic_a.initialize_population()

    # Evaluate the population
    genetic_a.evaluate_population()

    # Sort population
    genetic_a.sort_population_by_fitness()

    # Find average fitness and best_fit
    genetic_a.find_avg_fit()
    genetic_a.find_gen_best_fit()

    # Use tqdm to track overall completion
    gen = -1
    for _ in tqdm(range(max_gens)):
        gen += 1

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

        """
        Algorithm visualization
        """
        # Draw the best path so far, the optimum tour if there is one, pass info to renderer as text,
        # and check for pygame events
        if SETTINGS['Display']:
            text = [f"{gen+1} / {max_gens}", f"{genetic_a.best.fitness}", f"{genetic_a.avg_gen_fit}", f"{num_nodes}",
                    f"{pop_size}", f"{elite_rate}", f"{cross_rate}", f"{mutate_rate}"]
            if opt_tour:
                text.append(f"{opt_tour_ind.fitness}")
            render_window.draw_frame(genetic_a.best.genes, text, opt_tour)
            r.event_listen()

    """
    Start of data collection and visualization
    """
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

    # Listen for user input to quit program
    if SETTINGS['Display']:
        while True:
            r.event_listen()


if __name__ == '__main__':
    main()
