import numpy as np
import matplotlib.pyplot as plt
from TravellingSalesman import tsp
from TravellingSalesman import divideconquer

def main():
    parser = tsp.TSPParser()
    instances = parser.get_instances()
    instances = sorted(instances, key=lambda x: x.best_path_length)

    labels = []
    opt_lengths = []
    calc_lengths = []
    percentage_diffs = []

    for instance in instances:
        labels.append(instance.name)
        opt_length = instance.best_path_length
        opt_lengths.append(opt_length)

        # Run divide and conquer algorithm
        divide_c = divideconquer.DC(instance.coords, instance.distance_matrix)
        solution = divide_c.run()
        calc_length = tsp.calculate_path_length(solution, instance.distance_matrix)
        calc_lengths.append(calc_length)

        # Calculate percentage difference
        if opt_length > 0:
            percentage_diff = ((calc_length - opt_length) / opt_length) * 100
        else:
            percentage_diff = 0  # Handle division by zero edge case

        percentage_diffs.append(percentage_diff)

        print(f"{instance.name}: DC length: {calc_length}, Opt length: {opt_length}, Percentage Diff: {percentage_diff:.2f}%")

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, opt_lengths, width, label='Expected')
    rects2 = ax.bar(x + width/2, calc_lengths, width, label='Actual')

    # Add percentage difference annotations
    for i, (rect1, rect2) in enumerate(zip(rects1, rects2)):
        diff = percentage_diffs[i]
        if diff > 0:
            ax.annotate(f'+{diff:.1f}%', xy=(rect2.get_x() + rect2.get_width() / 2, rect2.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        elif diff < 0:
            ax.annotate(f'{diff:.1f}%', xy=(rect2.get_x() + rect2.get_width() / 2, rect2.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Instance')
    ax.set_ylabel('Cost')
    ax.set_title('Divide and Conquer Solution Quality')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()

    # Adjust y-axis limits with some padding
    max_y = max(max(opt_lengths), max(calc_lengths)) * 1.1  # 10% padding
    ax.set_ylim([0, max_y])

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
