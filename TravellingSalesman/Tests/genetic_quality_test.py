import numpy as np
import matplotlib.pyplot as plt
from TravellingSalesman import tsp
from TravellingSalesman import genetic

SETTINGS = {
    'Pop Size': 100,
    'Crossover Rate': 1,
    'Mutation Rate': 1,
    'Elite Rate': 0.1,
    'Max Gens': 100,
    'Batch Size': 25
}

def main():
    parser = tsp.TSPParser()
    instances = parser.get_instances()
    instances = sorted(instances, key=lambda x: x.best_path_length)

    labels = []
    opt_costs = []
    avg_ga_costs = []

    # Calculate optimal costs once per instance
    for instance in instances:
        labels.append(instance.name)
        opt_costs.append(instance.best_path_length)

    # Calculate average GA solution quality for each instance
    for instance in instances:
        total_ga_cost = 0
        for _ in range(SETTINGS['Batch Size']):
            ga = genetic.GA(instance.num_nodes, SETTINGS['Pop Size'], SETTINGS['Crossover Rate'],
                                      SETTINGS['Mutation Rate'], instance.distance_matrix, SETTINGS['Elite Rate'],
                                      SETTINGS['Max Gens'])

            solution = ga.run()
            ga_cost = solution[0]
            total_ga_cost += ga_cost

        avg_ga_cost = total_ga_cost / SETTINGS['Batch Size']
        avg_ga_costs.append(avg_ga_cost)

        print(f"{instance.name}: Optimal Cost: {instance.best_path_length}, Avg GA Cost: {avg_ga_cost}")

    # Calculate percentage differences
    percentage_diffs = [(avg_ga_cost - opt_cost) / opt_cost * 100 for avg_ga_cost, opt_cost in zip(avg_ga_costs, opt_costs)]
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, opt_costs, width, label='Optimal Cost')
    rects2 = ax.bar(x + width/2, avg_ga_costs, width, label='Avg GA Cost')

    # Add percentage difference labels only for GA costs
    def autolabel(rects, percentage_diffs):
        for i, rect in enumerate(rects):
            if rects is rects2:  # Only label GA costs
                height = rect.get_height()
                diff = percentage_diffs[i]
                ax.annotate(f'{diff:.2f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

    autolabel(rects2, percentage_diffs)

    ax.set_xlabel('Instance')
    ax.set_ylabel('Cost')
    ax.set_title('Genetic Algorithm Solution Quality | Batch Size = 25 | Max Gens = 100 | Pop Size = 100 | Mutation '
                 'Rate = 1 | Crossover Rate = 1 | Elite Rate = 0.1')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
