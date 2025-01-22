import numpy as np
import random

class JobShopGA:
    def __init__(self, jobs, num_machines, pop_size=50, generations=100, crossover_prob=0.8, mutation_prob=0.2):
        self.jobs = jobs
        self.num_machines = num_machines
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            chromosome = []
            for job_id, job in enumerate(self.jobs):
                chromosome.extend([(job_id, op_id) for op_id in range(len(job))])
            random.shuffle(chromosome)
            population.append(chromosome)
        return population

    def decode_chromosome(self, chromosome):
        schedule = {m: [] for m in range(self.num_machines)}
        job_times = {j: 0 for j in range(len(self.jobs))}
        machine_times = {m: 0 for m in range(self.num_machines)}
        
        for job_id, op_id in chromosome:
            machine, duration = self.jobs[job_id][op_id]
            start_time = max(job_times[job_id], machine_times[machine])
            job_times[job_id] = start_time + duration
            machine_times[machine] = start_time + duration
            schedule[machine].append((job_id, op_id, start_time, duration))
        
        return schedule, max(machine_times.values())

    def fitness(self, chromosome):
        _, makespan = self.decode_chromosome(chromosome)
        return -makespan  # Negative because we minimize makespan

    def selection(self):
        tournament_size = 5
        selected = random.sample(self.population, tournament_size)
        selected.sort(key=lambda x: self.fitness(x), reverse=True)
        return selected[0]

    def crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child1, child2 = parent1[:], parent2[:]
        
        def pmx_cross(p1, p2):
            mapping = {}
            for i in range(start, end + 1):
                mapping[p1[i]] = p2[i]
                mapping[p2[i]] = p1[i]
            
            def apply_mapping(seq):
                return [mapping.get(gene, gene) for gene in seq]
            
            return apply_mapping(p1[:start]) + p2[start:end+1] + apply_mapping(p1[end+1:])
        
        return pmx_cross(parent1, parent2), pmx_cross(parent2, parent1)

    def mutate(self, chromosome):
        idx1, idx2 = random.sample(range(len(chromosome)), 2)
        chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
        return chromosome

    def evolve(self):
        for _ in range(self.generations):
            new_population = []
            self.population.sort(key=lambda x: self.fitness(x), reverse=True)
            new_population.append(self.population[0])  # Elitism
            
            while len(new_population) < self.pop_size:
                parent1, parent2 = self.selection(), self.selection()
                
                if random.random() < self.crossover_prob:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1[:], parent2[:]
                
                if random.random() < self.mutation_prob:
                    child1 = self.mutate(child1)
                if random.random() < self.mutation_prob:
                    child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            self.population = new_population[:self.pop_size]
        
        best_solution = max(self.population, key=lambda x: self.fitness(x))
        return self.decode_chromosome(best_solution)

def parse_dataset(data):
    lines = data.strip().split("\n")
    num_jobs, num_machines = map(int, lines[0].split())
    jobs = []
    for line in lines[1:num_jobs+1]:
        job_data = list(map(int, line.split()))
        jobs.append([(job_data[i], job_data[i+1]) for i in range(0, len(job_data), 2)])
    return jobs, num_machines

# Example dataset
raw_data = """10 10
4 88 8 68 6 94 5 99 1 67 2 89 9 77 7 99 0 86 3 92
5 72 3 50 6 69 4 75 2 94 8 66 0 92 1 82 7 94 9 63
9 83 8 61 0 83 1 65 6 64 5 85 7 78 4 85 2 55 3 77
7 94 2 68 1 61 4 99 3 54 6 75 5 66 0 76 9 63 8 67
3 69 4 88 9 82 8 95 0 99 2 67 6 95 5 68 7 67 1 86
1 99 4 81 5 64 6 66 8 80 2 80 7 69 9 62 3 79 0 88
7 50 1 86 4 97 3 96 0 95 8 97 2 66 5 99 6 52 9 71
4 98 6 73 3 82 2 51 1 71 5 94 7 85 0 62 8 95 9 79
0 94 6 71 3 81 7 85 1 66 2 90 4 76 5 58 8 93 9 97
3 50 0 59 1 82 8 67 7 56 9 96 6 58 4 81 5 59 2 96"""

jobs, num_machines = parse_dataset(raw_data)
ga = JobShopGA(jobs, num_machines)
schedule, best_time = ga.evolve()
print("Best Schedule:", schedule)
print("Best Makespan:", best_time)
