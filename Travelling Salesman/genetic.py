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
    n = np.random.randint(1, len(offspring.genes) - 1)
    m = np.random.randint(1, len(offspring.genes) - 1)
    if n > m:
        n, m = m, n
    if m - n < 1:
        if n == 1:
            m += 1
        elif m == len(offspring.genes) - 1:
            n -= 1
    tmp = offspring.genes[n]
    del offspring.genes[n]
    offspring.genes.insert(m, tmp)


def inversion_mutation(offspring):
    n = np.random.randint(1, len(offspring.genes) - 1)
    m = np.random.randint(1, len(offspring.genes) - 1)
    if n > m:
        n, m = m, n
    if m - n < 2:
        if n == 1:
            m += 2
        elif m == len(offspring.genes) - 1:
            n -= 2
    offspring.genes[n:m].reverse()


def swap_mutation(offspring):
    n = np.random.randint(1, len(offspring.genes) - 1)
    m = np.random.randint(1, len(offspring.genes) - 1)
    if n > m:
        n, m = m, n
    if m - n < 1:
        if n == 1:
            m += 1
        elif m == len(offspring.genes) - 1:
            n -= 1
    offspring.genes[n], offspring.genes[m] = offspring.genes[m], offspring.genes[n]


def shuffle_mutation(offspring):
    n = np.random.randint(1, len(offspring.genes) - 1)
    m = np.random.randint(1, len(offspring.genes) - 1)
    if n > m:
        n, m = m, n
    if m - n < 2:
        if n == 1:
            m += 2
        elif m == len(offspring.genes) - 1:
            n -= 2

    np.random.shuffle(offspring.genes[n:m])


def displacement_mutation(offspring):
    n = np.random.randint(1, len(offspring.genes) - 1)
    m = np.random.randint(1, len(offspring.genes) - 1)
    if n > m:
        n, m = m, n
    if m - n < 2:
        if n == 1:
            m += 2
        elif m == len(offspring.genes) - 1:
            n -= 2

    tmp = offspring.genes[n:m]
    del offspring.genes[n:m]

    s = np.random.randint(1, len(offspring.genes) - 1)
    for i in range(len(tmp)):
        offspring.genes.insert(s, tmp[i])
        s += 1


class GA(object):
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
        available_nodes = [i for i in range(0, self.num_nodes)]
        available_nodes.remove(self.start_ind)
        if self.start_ind != self.end_ind:
            available_nodes.remove(self.end_ind)

        for i in range(self.population_size):
            new_genes = list(np.random.choice(available_nodes, len(available_nodes), replace=False))
            new_genes.insert(0, self.start_ind)
            new_genes.append(self.end_ind)
            new_individual = Individual(new_genes, np.inf)
            self.population.append(new_individual)
        self.generations.append(self.population)

    def evaluate_individual(self, ind):
        fit = 0
        for i in range(len(ind.genes) - 1):
            fit += self.distance_matrix[ind.genes[i]][ind.genes[i + 1]]
        ind.fitness = fit
        if ind.fitness < self.best.fitness:
            self.best = ind

    def evaluate_population(self, population=None):
        if population is None:
            for i in self.population:
                self.evaluate_individual(i)
        else:
            for j in population:
                self.evaluate_individual(j)
            return population

    def sort_population_by_fitness(self, pop=None):
        if pop is None:
            self.population = sorted(self.population, key=lambda x: x.fitness)
        else:
            pop = sorted(pop, key=lambda x: x.fitness)
            return pop

    def find_avg_fit(self):
        s = 0
        for i in self.population:
            s += i.fitness
        s /= self.population_size
        self.avg_gen_fits.append(s)
        self.avg_gen_fit = s

    def find_gen_best_fit(self):
        self.gen_best_fits.append(self.best.fitness)

    def find_parent_probabilities(self):
        inverse_fits = []
        probabilities = []
        for i in self.population:
            inverse_fits.append(1 / np.exp(np.sqrt(i.fitness)))
        s = sum(inverse_fits)
        for i in inverse_fits:
            probabilities.append(i / s)
        return probabilities

    def select_parents(self, probs):
        parents = [None, None]
        parents[0] = np.random.choice(self.population, p=probs)
        parents[1] = np.random.choice(self.population, p=probs)
        return parents

    def mutate_batch(self, offsprings):
        # muts = 0
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
        # print(f"Mutations occurred: {muts} / {len(offsprings)*(len(self.population[0].genes))}")
        return offsprings

    def insertion_crossover(self, parents):
        p1 = parents[0].genes[1:-1]
        p2 = parents[1].genes[1:-1]
        r = np.random.randint(0, len(p1))
        new_genes = p1[:r]
        for gene in p2:
            if gene not in new_genes:
                new_genes.append(gene)
        new_genes.insert(0, self.start_ind)
        new_genes.append(self.end_ind)
        return Individual(new_genes, np.inf)

    def pmx_crossover(self, parents):
        """Performs Partially Mapped Crossover (PMX) on two parent lists of alleles."""
        parent1 = parents[0].genes[1:-1]
        parent2 = parents[1].genes[1:-1]

        # Select two random crossover points
        size = len(parent1)
        cxpoint1 = np.random.randint(0, size)
        cxpoint2 = np.random.randint(0, size)
        if cxpoint2 < cxpoint1:
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        # Initialize child with a copy of the first parent
        child = [-1] * size

        # Copy the crossover segment from the first parent to the child
        child[cxpoint1:cxpoint2 + 1] = parent1[cxpoint1:cxpoint2 + 1]

        # Map elements from the second parent to the child
        for i in range(cxpoint1, cxpoint2 + 1):
            if parent2[i] not in child:
                index = parent2.index(parent1[i])
                while cxpoint1 <= index <= cxpoint2:
                    index = parent2.index(parent1[index])
                child[index] = parent2[i]

        # Fill in the remaining elements from the second parent
        for i in range(size):
            if child[i] == -1:
                child[i] = parent2[i]

        child.insert(0, self.start_ind)
        child.append(self.end_ind)

        return Individual(child, np.inf)

    def generate_offspring(self):
        probs = self.find_parent_probabilities()
        offsprings = []
        for i in range(self.population_size):
            parents = self.select_parents(probs)
            p1 = parents[0]
            p2 = parents[1]
            r = np.random.rand()
            if r <= self.crossover_rate:
                offsprings.append(self.pmx_crossover((p1, p2)))
                offsprings.append(self.pmx_crossover((p2, p1)))
        return offsprings

    def select_new_population(self, offsprings):
        new_pop = [self.best]
        s = int(self.elite_rate*self.population_size)
        for i in range(s-1):
            new_pop.append(self.population[i])
        for i in range(self.population_size - s):
            if i >= len(offsprings):
                new_pop.append(self.population[i + s])
            else:
                new_pop.append(offsprings[i])

        self.population = new_pop[:]
        self.generations.append(self.population)

        self.find_gen_best_fit()
        self.find_avg_fit()
