"""
Script for running quality test on solutions from DC
"""
import numpy as np
import time
from matplotlib import pyplot as plt
from tqdm import tqdm

from TravellingSalesman import tsp
from TravellingSalesman import divideconquer


def main():
    parser = tsp.TSPParser()
    instances = parser.get_instances()
    for instance in instances:
        print(f"Instance Name: {instance.name}")
        print(f"Space Dimensions: {instance.dimensions}")
        print(f"Nodes: {instance.nodes}")
        print(f"Coordinates: {instance.coords}")
        print(f"Best Path: {instance.best_path}")
        print("--------------------")


if __name__ == '__main__':
    main()
