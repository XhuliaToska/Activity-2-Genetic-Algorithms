import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Define parameters
POPULATION_SIZE = 500  # Chosen based on problem complexity and balance between diversity and computational cost
BASE_MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.8
GENERATIONS = 500
ELITISM = True
TOURNAMENT_SIZE = 3

# Example Job Shop Problem (each job has a sequence of (machine, processing time))
JOBS = [
    [(0, 3), (1, 2), (2, 2)],
    [(0, 2), (2, 1), (1, 4)],
    [(1, 4), (2, 3)]
]

NUM_JOBS = len(JOBS)
NUM_MACHINES = max(max(op[0] for op in job) for job in JOBS) + 1

# --- CHROMOSOME REPRESENTATION ---
# A chromosome is represented as a permutation of job operations (job ID with operation index)

def create_chromosome() -> List[Tuple[int, int]]:
    """Creates a valid chromosome by randomly permuting job operations."""
    chromosome = []
    for job_id, job in enumerate(JOBS):
        for op_id in range(len(job)):
            chromosome.append((job_id, op_id))
    random.shuffle(chromosome)
    return chromosome

# --- FITNESS FUNCTION ---
def evaluate(chromosome: List[Tuple[int, int]]) -> int:
    machine_times = [0] * NUM_MACHINES
    job_times = [0] * NUM_JOBS
    
    for job_id, op_id in chromosome:
        machine, duration = JOBS[job_id][op_id]
        
        if op_id == 0:
            start_time = machine_times[machine]  
        else:
            start_time = max(machine_times[machine], job_times[job_id])
        
        end_time = start_time + duration
        machine_times[machine] = end_time
        job_times[job_id] = end_time
    
    return max(machine_times)

# --- SELECTION METHODS ---
def tournament_selection(population: List[List[Tuple[int, int]]], fitness: List[int]) -> List[Tuple[int, int]]:
    tournament = random.sample(list(zip(population, fitness)), TOURNAMENT_SIZE)
    return min(tournament, key=lambda x: x[1])[0]  

def stochastic_universal_sampling(population: List[List[Tuple[int, int]]], fitness: List[int]) -> List[Tuple[int, int]]:
    total_fitness = sum(1 / f for f in fitness)
    step_size = total_fitness / TOURNAMENT_SIZE
    start_point = random.uniform(0, step_size)
    pointers = [start_point + i * step_size for i in range(TOURNAMENT_SIZE)]
    selected = []
    
    current_fitness_sum = 0
    index = 0
    for ptr in pointers:
        while current_fitness_sum < ptr:
            current_fitness_sum += 1 / fitness[index]
            index += 1
        selected.append(population[index - 1])
    return random.choice(selected)  

# --- CROSSOVER METHODS ---
def edge_recombination_crossover(parent1: List[Tuple[int, int]], parent2: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    size = len(parent1)
    adjacency_list = {gene: set() for gene in set(parent1 + parent2)}

    for p in [parent1, parent2]:
        for i in range(len(p) - 1):
            adjacency_list[p[i]].add(p[i + 1])
            adjacency_list[p[i + 1]].add(p[i])
    child = []
    current = random.choice(parent1)
    while len(child) < size:
        child.append(current)
        for key in adjacency_list:
            adjacency_list[key].discard(current)
        if adjacency_list[current]:
            current = min(adjacency_list[current], key=lambda k: len(adjacency_list[k]))
        else:
            remaining = [g for g in parent1 if g not in child]
            if remaining:
                current = random.choice(remaining)
            else:
                break  

    return child

def uniform_crossover(parent1: List[Tuple[int, int]], parent2: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return [random.choice(gene_pair) for gene_pair in zip(parent1, parent2)]

# --- MUTATION METHODS ---
def swap_mutation(chromosome: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if len(chromosome) < 2:
        return chromosome  
    chromosome = chromosome[:] 
    a, b = random.sample(range(len(chromosome)), 2)
    chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
    return chromosome  
    a, b = random.sample(range(len(chromosome)), 2)
    chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
    return chromosome

def insertion_mutation(chromosome: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if len(chromosome) < 2:
        return chromosome
    chromosome = chromosome[:]  # Avoid modifying in place
    index = random.randint(0, len(chromosome) - 1)
    gene = chromosome.pop(index)
    insert_pos = random.randint(0, len(chromosome))
    chromosome.insert(insert_pos, gene)
    return chromosome

# --- EVOLUTIONARY PROCESS ---
def genetic_algorithm():
    population = [create_chromosome() for _ in range(POPULATION_SIZE)]
    best_solution = None
    best_fitness = float('inf')
    stagnation_counter = 0  
    
    for generation in range(GENERATIONS):
        fitness = [evaluate(chrom) for chrom in population]
        sorted_population = [ch for _, ch in sorted(zip(fitness, population))]
        
        if ELITISM:
            new_population = [sorted_population[0]]  
        else:
            new_population = []
        
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitness)
            parent2 = stochastic_universal_sampling(population, fitness)
            
            crossover_method = edge_recombination_crossover if generation % 2 == 0 else uniform_crossover
            if random.random() < CROSSOVER_RATE:
                child = crossover_method(parent1, parent2)
            else:
                child = parent1[:]
            
            mutation_rate = BASE_MUTATION_RATE + (0.2 if stagnation_counter > 20 else 0)
            if random.random() < mutation_rate:
                child = swap_mutation(child) if random.random() < 0.3 else insertion_mutation(child)
            
            new_population.append(child)
        
        population = new_population
        best_idx = fitness.index(min(fitness))
        if fitness[best_idx] < best_fitness:
            best_fitness = fitness[best_idx]
            best_solution = population[best_idx]
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        fitness_history.append(best_fitness)
        print(f'Generation {generation}: Best Makespan = {best_fitness}')
        
        if stagnation_counter > 100:  # Stop if no improvement for 50 generations
            print("Stopping early due to stagnation.")
            break
    
    plt.plot(fitness_history)
    plt.xlabel('Generation')
    plt.ylabel('Best Makespan')
    plt.title('Evolution of Best Makespan Over Generations')
    plt.show()
    
    return best_solution, best_fitness

# Run the genetic algorithm
best_solution, best_fitness = genetic_algorithm()
print("Best solution:", best_solution)
print("Best makespan:", best_fitness)
