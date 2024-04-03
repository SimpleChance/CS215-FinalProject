import random


def two_opt_mutation(tour):
    """Apply 2-opt mutation to the given tour."""
    # Randomly select two distinct indices in the tour
    i, j = sorted(random.sample(range(len(tour)), 2))

    # Reverse the segment between the selected indices
    new_tour = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]

    return new_tour


def three_opt_mutation(tour):
    """Apply modified 3-opt mutation to the given tour."""
    # Randomly select three distinct indices in the tour
    i, j, k = sorted(random.sample(range(len(tour)), 3))

    # Ensure the selected swath has at least 2 elements
    while k - i < 2:
        i, j, k = sorted(random.sample(range(len(tour)), 3))

    # Possible 3-opt moves
    moves = [
        tour[:i] + tour[j:k + 1] + tour[i:j] + tour[k + 1:],
        tour[:i] + tour[j:k + 1][::-1] + tour[i:j] + tour[k + 1:],
        tour[:i] + tour[k:j - 1:-1] + tour[i:k + 1] + tour[j:],
        tour[:i] + tour[k:j - 1:-1] + tour[i:k + 1][::-1] + tour[j:]
    ]

    # Choose a random move
    new_tour = random.choice(moves)

    return new_tour


# Example usage:
tour = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("Original tour:", tour)

# Applying 2-opt mutation
new_tour_2opt = two_opt_mutation(tour)
print("After 2-opt mutation:", new_tour_2opt)

# Applying 3-opt mutation
new_tour_3opt = three_opt_mutation(tour)
print("After 3-opt mutation:", new_tour_3opt)