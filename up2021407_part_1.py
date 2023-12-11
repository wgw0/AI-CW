import random

def create_population(size, initial_chromosome, mutation_rate):
    # Creates the initial population from the initial chromosome.
    # Mutates the initial chromosome to create a varied initial population.
    
    return [mutate(initial_chromosome, mutation_rate) for _ in range(size)]

def fitness(chromosome, target):
    # Fitness function.
    # Compares the chromosome to the target and counts the number of matching bits.
    
    return sum(chromosome_bit == target_bit for chromosome_bit, target_bit in zip(chromosome, target))

def select(population, target):
    # Selection process.
    # Selects two parents with probabilities proportional to their fitness scores.
    
    fitness_scores = [fitness(chromosome, target) for chromosome in population]
    total_fitness = sum(fitness_scores)
    selection_probs = [score / total_fitness for score in fitness_scores]
    return random.choices(population, weights=selection_probs, k=2)

def crossover(parent1, parent2):
    # Crossover operation.
    # Creates a child by combining parts of both parents' chromosomes.
    
    crossover_point = random.randint(1, len(parent1) - 1)
    return parent1[:crossover_point] + parent2[crossover_point:]

def mutate(chromosome, mutation_rate):
    # Mutation operation.
    # Flips random bits in the chromosome based on the mutation rate.
    
    return ''.join(bit if random.random() > mutation_rate else '0' if bit == '1' else '1' for bit in chromosome)

def genetic_algorithm(target, initial_chromosome, population_size=100, mutation_rate=0.01, max_generations=1000):
    # Main Genetic Algorithm function.
    # Evolves the population over several generations to reach the target chromosome.

    population = create_population(population_size, initial_chromosome, mutation_rate)

    for generation in range(max_generations):
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select(population, target)
            child1 = mutate(crossover(parent1, parent2), mutation_rate)
            child2 = mutate(crossover(parent2, parent1), mutation_rate)
            new_population.extend([child1, child2])
        population = new_population

        # Calculate fitness for the best chromosome in this generation
        best_chromosome = max(population, key=lambda chromosome: fitness(chromosome, target))
        best_fitness = fitness(best_chromosome, target)
        
        # Print the fitness for the best chromosome in this generation
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

        # Check for the optimal solution
        if any(fitness(chromosome, target) == len(target) for chromosome in population):
            print(f"Optimal solution found in generation {generation}")
            break
    else:
        print("Optimal solution not found")

# Parameters
initial_chromosome = '10010111001010010011000000011010'
target_chromosome = '11111111111111111111111111111111'

# Run the Genetic Algorithm
genetic_algorithm(target_chromosome, initial_chromosome)