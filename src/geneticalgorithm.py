import random
import numpy as np
from typing import List, Tuple

# Define parameters
POPULATION_SIZE = 100  # Chosen based on problem complexity and balance between diversity and computational cost
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
GENERATIONS = 500
ELITISM = True
TOURNAMENT_SIZE = 5

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
    """Computes the makespan (total completion time) of the schedule."""
    machine_times = [0] * NUM_MACHINES
    job_times = [0] * NUM_JOBS
    
    for job_id, op_id in chromosome:
        machine, duration = JOBS[job_id][op_id]
        start_time = max(machine_times[machine], job_times[job_id])
        end_time = start_time + duration
        machine_times[machine] = end_time
        job_times[job_id] = end_time
    
    return max(machine_times)

# --- SELECTION METHODS ---
def tournament_selection(population: List[List[Tuple[int, int]]], fitness: List[int]) -> List[Tuple[int, int]]:
    """Selects a parent using tournament selection."""
    tournament = random.sample(list(zip(population, fitness)), TOURNAMENT_SIZE)
    return min(tournament, key=lambda x: x[1])[0]  # Return chromosome with best fitness

def roulette_wheel_selection(population: List[List[Tuple[int, int]]], fitness: List[int]) -> List[Tuple[int, int]]:
    """Roulette wheel selection proportional to fitness (inverse of makespan)."""
    total_fitness = sum(1 / f for f in fitness)
    pick = random.uniform(0, total_fitness)
    current = 0
    for chromo, fit in zip(population, fitness):
        current += 1 / fit
        if current > pick:
            return chromo
    return population[-1]  # In case of rounding errors

# --- CROSSOVER METHODS ---
def ordered_crossover(parent1: List[Tuple[int, int]], parent2: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Implements ordered crossover (OX) for permutation-based chromosomes."""
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = parent1[start:end]
    remaining = [gene for gene in parent2 if gene not in child]
    i, j = 0, 0
    while i < size:
        if child[i] is None:
            child[i] = remaining[j]
            j += 1
        i += 1
    return child

def uniform_crossover(parent1: List[Tuple[int, int]], parent2: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Uniform crossover where each gene is chosen randomly from either parent."""
    return [random.choice(gene_pair) for gene_pair in zip(parent1, parent2)]

# --- MUTATION METHODS ---
def swap_mutation(chromosome: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Swaps two randomly selected positions."""
    a, b = random.sample(range(len(chromosome)), 2)
    chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
    return chromosome

def inversion_mutation(chromosome: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Reverses a random subsequence."""
    start, end = sorted(random.sample(range(len(chromosome)), 2))
    chromosome[start:end] = reversed(chromosome[start:end])
    return chromosome

# --- EVOLUTIONARY PROCESS ---
def genetic_algorithm():
    population = [create_chromosome() for _ in range(POPULATION_SIZE)]
    best_solution = None
    best_fitness = float('inf')
    stagnation_counter = 0  # To track stationary state
    
    for generation in range(GENERATIONS):
        fitness = [evaluate(chrom) for chrom in population]
        sorted_population = [ch for _, ch in sorted(zip(fitness, population))]
        
        if ELITISM:
            new_population = [sorted_population[0]]  # Preserve the best solution
        else:
            new_population = []
        
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitness)
            parent2 = roulette_wheel_selection(population, fitness)
            
            if random.random() < CROSSOVER_RATE:
                child = ordered_crossover(parent1, parent2)
            else:
                child = parent1[:]
            
            if random.random() < MUTATION_RATE:
                child = swap_mutation(child) if random.random() < 0.5 else inversion_mutation(child)
            
            new_population.append(child)
        
        population = new_population
        best_idx = fitness.index(min(fitness))
        if fitness[best_idx] < best_fitness:
            best_fitness = fitness[best_idx]
            best_solution = population[best_idx]
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        print(f'Generation {generation}: Best Makespan = {best_fitness}')
        
        if stagnation_counter > 50:  # Stop if no improvement for 50 generations
            print("Stopping early due to stagnation.")
            break
    
    return best_solution, best_fitness

# Run the genetic algorithm
best_solution, best_fitness = genetic_algorithm()
print("Best solution:", best_solution)
print("Best makespan:", best_fitness)
