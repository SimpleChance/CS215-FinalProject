Chance Jewell, Cade Parlato, Jacob Gibson

This repository contains all files associated with our CS215 Final Project - Analysis of Brute Force, Divide and Conquer, and Genetic Algorithms for Solving TSP.

The time-complexity for each algorithm can be determined as follows:
- Brute Force:
  - Our brute force search method involves generating all the permutations of a given node list, and then exhaustively searching every permutation for the shortest path length.
  - Generating the permutations of n nodes takes O(n!) and scanning through those permutations takes O(n). Therefore, the time complexity for our brute force algorithm is O(n!), where n is the number of nodes.
- Genetic:
  - Our genetic algorithm involves iterating over a population and producing a new population based on the best individuals for a certain number of generations.
  - A single epoch (generation), performs crossover, mutates offspring, evaluates the offspring, sorts the offspring, and generates a new population.
  - Crossover, in our case PMX Crossover, operates in O(n) time, where n is the length of individual's genes (num nodes).
  - Mutation, in which we use many different mutation operations, operates in O(1) in the best case and O(n^2) in the worst case. Where n is the length of individual's genes (num nodes).
  - Evaluating the population operates in O(n*m) time, where n is the length of individual's genes (num nodes), and m is the population size.
  - Sorting the population operates in O(n*log(n)) time, where n is the population size.
  - Generating a new population operates in O(n) time, where n is the population size.
  - Therefore, the overall time complexity for our algorithm is:
    - O(s * {(n^2) + (n\*m) + (m*log(m)})

Class files, test files, and the main script for the TSP demo are under the Travelling Salesman folder.
Matplotlib figures from the test scripts are saved in the Test Results folder.
.opt.tsp and .tsp instance files are saved in the TSP Instances folder.

- tsp.py contains parsing and precompute methods for tsp instances ~ dependencies: numpy
- bruteforce.py contains the BF class and associated methods ~ dependencies: tqdm
- genetic.py contains the GA class and associated methods ~ dependencies: numpy and dataclasses
- render.py contains the Renderer class and associated methods ~ dependencies: pygame and sys
- main.py contains the entry point for the demo as well as data collection and analysis ~ dependencies: tdqm, numpy, time, matplotlib

main.py is where BF and GA objects are created, tracked, and analyzed. As well as where the node list and distances are precomputed.

test1_bruteforce.py is a script to collect time data for our brute force algorithm and display that data with matplotlib.

test1_genetic.py is a script to collect time data for our genetic algorithm based on a variable number of nodes.

test2_genetic.py is a script to collect time data for our genetic algorithm based on variable population size.

test3_genetic.py is a script to collect time data for our genetic algorithm based on variable maximum generations.

To install packages, cd into ...\CS215-FinalProject and paste the following into the terminal:
- pip install numpy
- pip install matplotlib
- pip install pygame
- pip install tqdm

To run the main demo, cd into ...\CS215-FinalProject\'Travelling Salesman' and paste the following into the terminal:
- ./main.py

TODO:
  - Divide and Conquer class file
