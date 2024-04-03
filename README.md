Chance Jewell, Cade Parlato, Jacob Gibson

This repository contains all files associated with our CS215 Final Project - Analysis of Brute Force, Divide and Conquer, and Genetic Algorithms for Solving TSP.

Class files and the main script for the TSP demo are under the Travelling Salesman folder.

- bruteforce.py contains the BF class and associated methods ~ dependencies: itertools
- genetic.py contains the GA class and associated methods ~ dependencies: numpy and dataclasses
- render.py contains the Renderer class and associated methods ~ dependencies: pygame and sys
- main.py contains the entry point for the demo as well as data collection and analysis ~ dependencies: tdqm, numpy, time, matplotlib

main.py is where BF and GA objects are created, tracked, and analyzed. As well as where the node list and distances are precomputed.

TODO:
  - Divide and Conquer class file
  - Refactor comments and code throughout all files
  - Polish visuals for demoing GA (pygame) and algorithm comparisons (matplotlib)
  - Change node generation from random coordinates to a known TSP instance
