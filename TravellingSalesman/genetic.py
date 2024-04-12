"""
Class file for genetic algorithm members and methods
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class Individual:
    genes: list
    fitness: float


def insertion_mutation(offspring):
    """
        Description: Performs insertion mutation on an offspring.
        Args: Individual
        Returns: Void
        Time Complexity: O(n) : n = length of genes (num nodes)
    """
    # Random chromosome to mutate and index to insert at
    s = np.random.randint(1, len(offspring.genes) - 1)
    m = np.random.randint(1, len(offspring.genes) - 1)

    # Handle edge cases and ensure m > s
    if s > m:
        s, m = m, s
    if m - s < 1:
        if s == 1:
            m += 1
        elif m == len(offspring.genes) - 1:
            s -= 1

    # Delete node from current position and insert at m
    tmp = offspring.genes[s]
    del offspring.genes[s]
    offspring.genes.insert(m, tmp)


def inversion_mutation(offspring):
    """
        Description: Performs inversion mutation on an offspring.
        Args: Individual
        Returns: Void
        Time Complexity: O(n) : n = length of genes to be mutated (m-s)
    """
    # Random indices to invert chromosomes between
    s = np.random.randint(1, len(offspring.genes) - 1)
    m = np.random.randint(1, len(offspring.genes) - 1)

    # Handle edge cases and ensure m > s
    if s > m:
        s, m = m, s
    if m - s < 2:
        if s == 1:
            m += 2
        elif m == len(offspring.genes) - 1:
            s -= 2

    # Invert chromosomes
    offspring.genes[s:m].reverse()


def swap_mutation(offspring):
    """
        Description: Performs swap mutation on an offspring.
        Args: Individual
        Returns: Void
        Time Complexity: O(1)
    """
    # Random chromosomes to swap
    s = np.random.randint(1, len(offspring.genes) - 1)
    m = np.random.randint(1, len(offspring.genes) - 1)

    # Handle edge cases and ensure m > s
    if s > m:
        s, m = m, s
    if m - s < 1:
        if s == 1:
            m += 1
        elif m == len(offspring.genes) - 1:
            s -= 1

    # Swap chromosomes
    offspring.genes[s], offspring.genes[m] = offspring.genes[m], offspring.genes[s]


def shuffle_mutation(offspring):
    """
        Description: Performs shuffle mutation on an offspring.
        Args: Individual
        Returns: Void
        Time Complexity: O(n) : n = length of genes to be mutated (m-s)
    """
    # Random indices to shuffle between
    s = np.random.randint(1, len(offspring.genes) - 1)
    m = np.random.randint(1, len(offspring.genes) - 1)

    # Handle edge cases and ensure m > s
    if s > m:
        s, m = m, s
    if m - s < 2:
        if s == 1:
            m += 2
        elif m == len(offspring.genes) - 1:
            s -= 2

    # Shuffle chromosomes
    np.random.shuffle(offspring.genes[s:m])


def displacement_mutation(offspring):
    """
        Description: Performs displacement mutation on an offspring.
        Args: Individual
        Returns: Void
        Time Complexity: O(n * m) : n = length of genes to be mutated (m-s)
                                  : m = length of genes (num nodes)
    """
    # Random indices for chromosomes to displace
    s = np.random.randint(1, len(offspring.genes) - 1)
    m = np.random.randint(1, len(offspring.genes) - 1)

    # Handle edge cases and ensure m > s
    if s > m:
        s, m = m, s
    if m - s < 2:
        if s == 1:
            m += 2
        elif m == len(offspring.genes) - 1:
            s -= 2

    # Delete chromosomes
    tmp = offspring.genes[s:m]
    del offspring.genes[s:m]

    # Insert the deleted chromosomes at a random position
    r = np.random.randint(1, len(offspring.genes))
    for i in range(len(tmp)):
        offspring.genes.insert(r, tmp[i])
        r += 1


class GA(object):
    """
        Description: GA object class
        Args: int, int, float, float, list[][], float, int
    """
    def __init__(self, num_nodes, population_size, crossover_rate, mutation_rate, distance_matrix, elite_rate, max_gens,
                 start_ind=0, end_ind=0):
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_rate = elite_rate
        self.num_nodes = num_nodes
        self.distance_matrix = distance_matrix

        self.start_ind = start_ind
        self.end_ind = end_ind

        self.generations = []
        self.max_gens = max_gens
        self.avg_gen_fits = []
        self.avg_gen_fit = 0
        self.gen_best_fits = []
        self.population = []
        self.offspring_list = []
        self.parent_probabilities = []

        self.best = Individual([0 for _ in range(num_nodes)], np.inf)

    def initialize_population(self):
        """
            Description: Initializes a population for GA
            Args: None
            Returns: void
            Time Complexity: O(m*n) : n = num nodes, m = population size
        """
        # Create list of available nodes to choose from for individual genes
        available_nodes = [i for i in range(0, self.num_nodes)]
        available_nodes.remove(self.start_ind)
        if self.start_ind != self.end_ind:
            available_nodes.remove(self.end_ind)

        # Create a new individual with random genes and add it to the population
        for i in range(self.population_size):
            new_genes = list(np.random.choice(available_nodes, len(available_nodes), replace=False))
            new_genes.insert(0, self.start_ind)
            new_genes.append(self.end_ind)
            new_individual = Individual(new_genes, np.inf)
            self.population.append(new_individual)
        self.generations.append(self.population)

    def evaluate_individual(self, ind, opt_tour=False):
        """
            Description: Evaluates and assigns the fitness of an individual.
                         Updates the best individual found for the current generation.
            Args: Individual
            Returns: void
            Time Complexity: O(n) : n = length of genes (num nodes)
        """
        fit = 0
        for i in range(len(ind.genes) - 1):
            fit += self.distance_matrix[ind.genes[i]][ind.genes[i + 1]]
        ind.fitness = fit
        if ind.fitness < self.best.fitness and not opt_tour:
            self.best = ind

    def evaluate_population(self, population=None):
        """
            Description: Helper function to evaluate each individual in a population.
            Args: list[Individual]
            Returns: list[Individual] - updated fitness values if population was specified
            Time Complexity: O(n*m) : n = length of genes (num nodes), m = length of self.population (population size)
        """
        if population is None:
            for i in self.population:
                self.evaluate_individual(i)
        else:
            for j in population:
                self.evaluate_individual(j)
            return population

    def sort_population_by_fitness(self, pop=None):
        """
            Description: Sorts a given population by fitness
            Args: list[Individual]
            Returns: list[Individual] - if specified
            Time Complexity: O(n*log(n)) : n = length of self.population (population size)
        """
        if pop is None:
            self.population = sorted(self.population, key=lambda x: x.fitness)
        else:
            pop = sorted(pop, key=lambda x: x.fitness)
            return pop

    def find_avg_fit(self):
        """
            Description: Updates the average fitness for the current population.
            Args: None
            Returns: void
            Time Complexity: O(n) for all cases : n = length of self.population (population size)
        """
        s = 0
        for i in self.population:
            s += i.fitness
        s /= self.population_size
        self.avg_gen_fits.append(s)
        self.avg_gen_fit = s

    def find_gen_best_fit(self):
        """
            Description: Updates the list of best individuals with the best individual of the current population.
            Args: Individual
            Returns: void
            Time Complexity: O(1)
            """
        self.gen_best_fits.append(self.best.fitness)

    def find_parent_probabilities(self):
        """
            Description: Calculates the probability for each individual in the current population to reproduce.
                         Based on the fitness of the individual.
            Args: None
            Returns: list[float]
            Time Complexity: O(n) for all cases : n = length of self.population (population size)
        """
        inverse_fits = []
        probabilities = []
        for i in self.population:
            inverse_fits.append(1 / np.exp(np.sqrt(i.fitness)))
        s = sum(inverse_fits)
        for i in inverse_fits:
            probabilities.append(i / s)
        return probabilities

    def select_parents(self, num, probs):
        """
            Description: Selects random parents to reproduce based on parent probabilities.
            Args: int, list[float]
            Returns: list[Individual]
            Time Complexity: O(n*m): n = length of probs (population size)
                                     m = number of parents (cross rate * population size)
        """
        parents = [None for _ in range(num)]

        for i in range(num):
            parents[i] = np.random.choice(self.population, p=probs)

        return parents

    def mutate_batch(self, offsprings):
        """
            Description: Performs a random mutation on each offspring in the given list.
            Args: list[Individual]
            Returns: list[Individual]
            Time Complexity: O(n)   for the best case : n = length of offsprings list (cross rate * population size)
                             O(n*m) for the worst case : m = length of offspring genes (num nodes)
        """
        # Selects a random mutation to be performed on each offspring in the offsprings list.
        for off in offsprings:
            q = np.random.rand()
            if q <= self.mutation_rate:
                p = np.random.rand()
                if p <= 1/5:
                    insertion_mutation(off)
                elif p <= 2/5:
                    inversion_mutation(off)
                elif p <= 3/5:
                    swap_mutation(off)
                elif p <= 4/5:
                    shuffle_mutation(off)
                else:
                    displacement_mutation(off)
        return offsprings

    def insertion_crossover(self, parents):
        """
            Description: Performs insertion crossover to generate an offspring.
            Args: list[Individual]
            Returns: Individual
            Time Complexity: O(n) : n = length of genes (num nodes)
        """
        # Parent genes
        p1 = parents[0].genes[1:-1]
        p2 = parents[1].genes[1:-1]

        # Random swath of genes from p1
        r = np.random.randint(0, len(p1))
        new_genes = p1[:r]

        # Fill in remaining genes from p2
        for gene in p2:
            if gene not in new_genes:
                new_genes.append(gene)
        
        new_genes.insert(0, self.start_ind)
        new_genes.append(self.end_ind)
        return Individual(new_genes, np.inf)

    def pmx_crossover(self, parents):
        """
        Description: Performs Partially Mapped Crossover (PMX) to generate an offspring.
        Args: list[Individual]
        Returns: Individual
        Time Complexity: O(n) : n = length of parent genes (num nodes)
        """
        parent1 = parents[0].genes[1:-1]
        parent2 = parents[1].genes[1:-1]
        size = len(parent1)

        # Select two random crossover points
        cxpoint1 = np.random.randint(0, size)
        cxpoint2 = np.random.randint(0, size)
        if cxpoint2 < cxpoint1:
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        # Initialize child with a copy of the first parent
        child = parent1[:]

        # Create a mapping dictionary from the second parent
        mapping = {parent2[i]: i for i in range(cxpoint1, cxpoint2 + 1)}

        # Fill in the remaining elements from the second parent
        for i in range(size):
            if child[i] == -1:
                if parent2[i] in mapping:
                    child[i] = parent1[mapping[parent2[i]]]
                else:
                    child[i] = parent2[i]

        child.insert(0, self.start_ind)
        child.append(self.end_ind)
        return Individual(child, np.inf)

    def generate_offspring(self):
        """
            Description: Generates a batch of offspring based on parent probabilities and the crossover rate.
            Args: None
            Returns: list[Individual]
            Time Complexity: O(n*m) : n = length of parent genes (num nodes)
                                    : m = amount of population to generate offspring (cross rate * population size)
        """
        # Find the reproduction probabilities of the population
        probs = self.find_parent_probabilities()
        offsprings = []
        
        # Determine number of parents to reproduce
        s = int(self.crossover_rate * self.population_size)

        parents = self.select_parents(s, probs)
        
        # Perform crossover
        if s >= 2:
            for i in range(s-1):
                p1 = parents[i]
                p2 = parents[i+1]
                offsprings.append(self.pmx_crossover((p1, p2)))
                offsprings.append(self.pmx_crossover((p2, p1)))

        return offsprings

    def select_new_population(self, offsprings):
        """
            Description: Selects a new population given the new offspring.
                         Updates the average fitness and best fit individual for the current generation.
            Args: list[Individual
            Returns: void
            Time Complexity: O(n) : n = population size
        """
        # Add the all-time best solution to the new population
        new_pop = [self.best]
        
        # Determine number of individuals from previous generation to survive
        s = int(self.elite_rate*self.population_size)
        for i in range(s-1):
            new_pop.append(self.population[i])
        
        # Fill the remaining population slots with the best offspring
        for i in range(self.population_size - s):
            if i >= len(offsprings):    # If not enough offspring, individuals from the previous generation survive
                new_pop.append(self.population[i + s])  
            else:
                new_pop.append(offsprings[i])
        
        # Set the new population and add it the list of generations
        self.population = new_pop[:]
        self.generations.append(self.population)
        
        # Determine average fitness and best individual
        self.find_gen_best_fit()
        self.find_avg_fit()
