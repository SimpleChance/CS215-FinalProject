Chance Jewell, Cade Parlato, Jacob Gibson

This repository contains all files associated with our CS215 Final Project - Analysis of Brute Force, Divide and Conquer, and Genetic Algorithms for Solving TSP.

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
  - Run tests and save data for each algorithm
