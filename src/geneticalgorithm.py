import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Define parameters
POPULATION_SIZE = 200  
BASE_MUTATION_RATE = 0.4
mutation_rate = BASE_MUTATION_RATE
CROSSOVER_RATE = 0.9
GENERATIONS = 800
ELITISM = True
TOURNAMENT_SIZE = 3


JOBS = [
    [(9, 66), (5, 91), (4, 87), (2, 94), (7, 21), (3, 92), (1, 7), (0, 12), (8, 11), (6, 19)],
    [(3, 13), (2, 20), (4, 7), (1, 14), (9, 66), (0, 75), (6, 77), (5, 16), (7, 95), (8, 7)],
    [(8, 77), (7, 20), (2, 34), (0, 15), (9, 88), (5, 89), (6, 53), (3, 6), (1, 45), (4, 76)],
    [(3, 27), (2, 74), (6, 88), (4, 62), (7, 52), (8, 69), (5, 9), (9, 98), (0, 52), (1, 88)],
    [(4, 88), (6, 15), (1, 52), (2, 61), (7, 54), (0, 62), (8, 59), (5, 9), (3, 90), (9, 5)],
    [(6, 71), (0, 41), (4, 38), (3, 53), (7, 91), (8, 68), (1, 50), (5, 78), (2, 23), (9, 72)],
    [(3, 95), (9, 36), (6, 66), (5, 52), (0, 45), (8, 30), (4, 23), (2, 25), (7, 17), (1, 6)],
    [(4, 65), (1, 8), (8, 85), (0, 71), (7, 65), (6, 28), (5, 88), (3, 76), (9, 27), (2, 95)],
    [(9, 37), (1, 37), (4, 28), (3, 51), (8, 86), (2, 9), (6, 55), (0, 73), (7, 51), (5, 90)],
    [(3, 39), (2, 15), (6, 83), (9, 44), (7, 53), (0, 16), (4, 46), (5, 24), (1, 25), (8, 82)],
    [(1, 72), (4, 48), (0, 87), (2, 66), (9, 5), (6, 54), (7, 39), (8, 35), (5, 95), (3, 60)],
    [(1, 46), (3, 20), (0, 97), (2, 21), (9, 46), (7, 37), (8, 19), (4, 59), (6, 34), (5, 55)],
    [(5, 23), (3, 25), (6, 78), (1, 24), (0, 28), (7, 83), (8, 28), (9, 5), (2, 73), (4, 45)],
    [(1, 37), (0, 53), (7, 87), (4, 38), (3, 71), (5, 29), (9, 12), (8, 33), (6, 55), (2, 12)],
    [(4, 90), (8, 17), (2, 49), (3, 83), (1, 40), (6, 23), (7, 65), (9, 27), (5, 7), (0, 48)]
]
NUM_JOBS = len(JOBS)
NUM_MACHINES = max(max(op[0] for op in job) for job in JOBS) + 1

def create_chromosome() -> List[Tuple[int, int]]:
    chromosome = []
    for job_id, job in enumerate(JOBS):
        for op_id in range(len(job)):
            chromosome.append((job_id, op_id))
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

def tournament_selection(population: List[List[Tuple[int, int]]], fitness: List[int]) -> List[Tuple[int, int]]:
    selected = random.sample(list(zip(population, fitness)), TOURNAMENT_SIZE)
    return min(selected, key=lambda x: x[1])[0]

def roulette_wheel_selection(population: List[List[Tuple[int, int]]], fitness: List[int]) -> List[Tuple[int, int]]:
    total_fitness = sum(1 / f for f in fitness if f > 0)
    pick = random.uniform(0, total_fitness)
    current = 0
    for chrom, fit in zip(population, fitness):
        current += 1 / fit if fit > 0 else 0
        if current >= pick:
            return chrom
    return population[-1]

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
    """Uniform Crossover ensuring both parents are of the same length."""
    size = min(len(parent1), len(parent2))  # Prevents index out of range error
    child = [parent1[i] if random.random() < 0.5 else parent2[i] for i in range(size)]
    if len(parent1) > size:
        child.extend(parent1[size:])
    elif len(parent2) > size:
        child.extend(parent2[size:])
    return child
    size = len(parent1)
    child = [parent1[i] if random.random() < 0.5 else parent2[i] for i in range(size)]
    return child

def swap_mutation(chromosome: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if len(chromosome) < 2:
        return chromosome
    a, b = random.sample(range(len(chromosome)), 2)
    chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
    return chromosome

def inversion_mutation(chromosome: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if len(chromosome) < 3:
        return chromosome
    start, end = sorted(random.sample(range(len(chromosome)), 2))
    chromosome[start:end] = reversed(chromosome[start:end])
    return chromosome

def genetic_algorithm():
    global fitness_history
    population = [create_chromosome() for _ in range(POPULATION_SIZE)]
    best_solution = None
    best_fitness = float('inf')
    stagnation_counter = 0
    fitness_history = []
    for generation in range(GENERATIONS):
        fitness = [evaluate(chrom) for chrom in population]
        sorted_population = [ch for _, ch in sorted(zip(fitness, population))]
        if ELITISM:
            new_population = sorted_population[:max(1, int(POPULATION_SIZE * 0.05))]
        else:
            new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1 = roulette_wheel_selection(population, fitness)
            parent2 = tournament_selection(population, fitness)
            if random.random() < CROSSOVER_RATE:
                child = edge_recombination_crossover(parent1, parent2) if random.random() < 0.5 else uniform_crossover(parent1, parent2)
            else:
                child = parent1[:]
            if random.random() < mutation_rate:
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
        fitness_history.append(best_fitness)
        print(f'Generation {generation}: Best Makespan = {best_fitness}')
        if stagnation_counter > 50:
            print("Stopping early due to stagnation.")
            break
    plt.plot(fitness_history)
    plt.xlabel('Generation')
    plt.ylabel('Best Makespan')
    plt.title('Evolution of Best Makespan Over Generations')
    plt.show()
    return best_solution, best_fitness

best_solution, best_fitness = genetic_algorithm()
print("Best solution:", best_solution)
print("Best makespan:", best_fitness)
