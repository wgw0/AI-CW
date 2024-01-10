import random

def create_population(size, chromosome_length):
    # Creates an initial population of chromosomes.
    # Each chromosome is a random string of 0s and 1s of the specified length.
    return [''.join(random.choice('01') for _ in range(chromosome_length)) for _ in range(size)]

def fitness(chromosome, target):
    # Fitness function.
    # Compares each bit of the chromosome to the target and counts the number of matching bits.
    # The higher the count, the better the fitness.
    return sum(chromosome_bit == target_bit for chromosome_bit, target_bit in zip(chromosome, target))

def roulette_wheel_selection(population, fitness_scores):
    # Roulette wheel selection process.
    # Selects parents for the next generation with probabilities proportional to their fitness scores.
    # The higher the fitness score, the higher the chance of being selected.

    # Normalise the fitness scores to sum up to 1.
    total_fitness = sum(fitness_scores)
    normalised_fitness = [f / total_fitness for f in fitness_scores]

    # Calculate cumulative fitness for selection.
    cumulative_fitness = [sum(normalised_fitness[:i+1]) for i in range(len(normalised_fitness))]

    # Select parents based on the cumulative fitness.
    selected_parents = []
    for _ in range(len(population)):
        rand = random.random()
        for i, fitness_value in enumerate(cumulative_fitness):
            if rand <= fitness_value:
                selected_parents.append(population[i])
                break
    return selected_parents

def crossover(parent1, parent2):
    # Crossover operation.
    # Creates a child by combining parts of both parents' chromosomes at a random point.
    crossover_point = random.randint(1, len(parent1) - 1)
    return parent1[:crossover_point] + parent2[crossover_point:]

def mutate(chromosome, mutation_rate):
    # Mutation operation.
    # Flips random bits in the chromosome based on the mutation rate.
    # This introduces genetic diversity and helps prevent premature convergence.
    return ''.join(bit if random.random() > mutation_rate else '0' if bit == '1' else '1' for bit in chromosome)

def genetic_algorithm(target, population_size=100, chromosome_length=32, mutation_rate=0.01, max_generations=1000):
    # Main Genetic Algorithm function.
    # Evolves the population over several generations to reach the target chromosome.
    # Steps include creating an initial population, evaluating fitness, selecting parents, crossover, and mutation.

    # Create initial population.
    population = create_population(population_size, chromosome_length)
    # print("Initial Population:") THESE CAN BE UNCOMMENTED TO DEMONSTRATE THE INITIAL POP
    # print(population)

    for generation in range(max_generations):
        # Calculate fitness for each chromosome in the population.
        fitness_scores = [fitness(chromosome, target) for chromosome in population]

        # Perform roulette wheel selection.
        parents = roulette_wheel_selection(population, fitness_scores)

        # Generate a new population using crossover and mutation.
        new_population = []
        for i in range(0, population_size, 2):
            child1 = mutate(crossover(parents[i], parents[i + 1]), mutation_rate)
            child2 = mutate(crossover(parents[i + 1], parents[i]), mutation_rate)
            new_population.extend([child1, child2])
        population = new_population

        # Find and print the best fitness in this generation.
        best_chromosome = max(population, key=lambda chromosome: fitness(chromosome, target))
        best_fitness = fitness(best_chromosome, target)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

        # Check for the optimal solution.
        if any(fitness(chromosome, target) == chromosome_length for chromosome in population):
            print(f"Optimal solution found in generation {generation}")
            # print("Final Population:") THESE CAN BE UNCOMMENTED TO DEMONSTRATE THE FINAL POP
            # print(population)
            break
    else:
        print("Optimal solution not found")

# Define the target chromosome.
target_chromosome = '11111111111111111111111111111111'

# Run the Genetic Algorithm.
genetic_algorithm(target_chromosome)
