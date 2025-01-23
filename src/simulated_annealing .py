import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Define parameters
POPULATION_SIZE = 200  # Balanced between diversity and computational cost
BASE_MUTATION_RATE = 0.3
mutation_rate = BASE_MUTATION_RATE
CROSSOVER_RATE = 0.8
GENERATIONS = 500
ELITISM = True
TOURNAMENT_SIZE = 3
STAGNATION_LIMIT = 50

# Simulated Annealing Parameters
INITIAL_TEMPERATURE = 1000
COOLING_RATE = 0.993  # Adjusted cooling rate
MIN_TEMPERATURE = 1
ITERATIONS_PER_TEMPERATURE = 500

# Example Job Shop Problem
JOBS = [
    [(9, 66), (5, 91), (4, 87), (2, 94), (7, 21), (3, 92), (1, 7), (0, 12), (8, 11), (6, 19)],
    [(3, 13), (2, 20), (4, 7), (1, 14), (9, 66), (0, 75), (6, 77), (5, 16), (7, 95), (8, 7)],
    [(8, 77), (7, 20), (2, 34), (0, 15), (9, 88), (5, 89), (6, 53), (3, 6), (1, 45), (4, 76)],
    [(3, 27), (2, 74), (6, 88), (4, 62), (7, 52), (8, 69), (5, 9), (9, 98), (0, 52), (1, 88)],
    [(4, 88), (6, 15), (1, 52), (2, 61), (7, 54), (0, 62), (8, 59), (5, 9), (3, 90), (9, 5)]
]

NUM_JOBS = len(JOBS)
NUM_MACHINES = max(max(op[0] for op in job) for job in JOBS) + 1

def create_chromosome() -> List[Tuple[int, int]]:
    chromosome = [(job_id, op_id) for job_id, job in enumerate(JOBS) for op_id in range(len(job))]
    random.shuffle(chromosome)
    return chromosome

def evaluate(chromosome: List[Tuple[int, int]]) -> int:
    machine_times = [0] * NUM_MACHINES
    job_times = [0] * NUM_JOBS
    for job_id, op_id in chromosome:
        machine, duration = JOBS[job_id][op_id]
        start_time = max(machine_times[machine], job_times[job_id])
        end_time = start_time + duration
        machine_times[machine] = end_time
        job_times[job_id] = end_time
    return max(machine_times)

def order_crossover(parent1: List[Tuple[int, int]], parent2: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    size = len(parent1)
    child = [None] * size
    start, end = sorted(random.sample(range(size), 2))
    child[start:end] = parent1[start:end]
    fill_pos = end
    for gene in parent2:
        if gene not in child:
            while child[fill_pos] is not None:
                fill_pos = (fill_pos + 1) % size  # Ensure wrap-around
            child[fill_pos] = gene
    return child

def inversion_mutation(chromosome: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if len(chromosome) < 3:
        return chromosome
    start, end = sorted(random.sample(range(len(chromosome)), 2))
    chromosome[start:end] = reversed(chromosome[start:end])
    return chromosome

def tournament_selection(population: List[List[Tuple[int, int]]], fitness: List[int]) -> List[Tuple[int, int]]:
    selected = random.sample(list(zip(population, fitness)), TOURNAMENT_SIZE)
    return min(selected, key=lambda x: x[1])[0]

def genetic_algorithm():
    population = [create_chromosome() for _ in range(POPULATION_SIZE)]
    best_solution = None
    best_fitness = float('inf')
    stagnation_counter = 0
    
    for generation in range(GENERATIONS):
        fitness = [evaluate(chrom) for chrom in population]
        sorted_population = [ch for _, ch in sorted(zip(fitness, population))]
        
        if ELITISM:
            new_population = sorted_population[:max(1, int(POPULATION_SIZE * 0.05))]
        else:
            new_population = []
        
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitness)
            parent2 = tournament_selection(population, fitness)
            if random.random() < CROSSOVER_RATE:
                child = order_crossover(parent1, parent2)
            else:
                child = parent1[:]
            if random.random() < mutation_rate:
                child = inversion_mutation(child)
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
        
        if stagnation_counter > STAGNATION_LIMIT:
            print("Stopping early due to stagnation.")
            break
    
    return best_solution, best_fitness

def simulated_annealing():
    current_solution = create_chromosome()
    current_cost = evaluate(current_solution)
    best_solution, best_cost = current_solution, current_cost
    temperature = INITIAL_TEMPERATURE
    
    while temperature > MIN_TEMPERATURE:
        for _ in range(ITERATIONS_PER_TEMPERATURE):
            neighbor = inversion_mutation(current_solution[:])
            neighbor_cost = evaluate(neighbor)
            
            if neighbor_cost < current_cost or random.random() < np.exp(-(neighbor_cost - current_cost) / temperature):
                current_solution, current_cost = neighbor, neighbor_cost
            
            if current_cost < best_cost:
                best_solution, best_cost = current_solution, current_cost
        
        temperature *= COOLING_RATE
    
    return best_solution, best_cost

best_ga_solution, best_ga_fitness = genetic_algorithm()
best_sa_solution, best_sa_fitness = simulated_annealing()
print("Genetic Algorithm Best Makespan:", best_ga_fitness)
print("Simulated Annealing Best Makespan:", best_sa_fitness)