from os import stat
from genetic import *



def test_Al10():
    molecule = Molecule.load('al_n.inp', rand_range=5)
    mutate_methods = Mutate([Mutate.swap_mutate, Mutate.mutate_angles, Mutate.mutate_distances], [1, 1, 1])
    crossover_methods = Crossover([Crossover.crossover_n, Crossover.crossover_1, Crossover.crossover_2], [1, 1, 1])
    create_methods = Create([Create.randomize, Create.mutate_first], [1, 1])
    strategies = Strategies([create_methods, mutate_methods, crossover_methods], [0, 1, 1])
    max_age = 10
    pool_size = 5
    max_seconds = 3600 * 10
    time_tolerance = 3600
    crossover_elitism = None
    fitness_param = '!RKS STATE 1.1 Energy'

    result, strategies = optimize(molecule, fitness_param, strategies, max_age, pool_size, max_seconds, time_tolerance, 
    crossover_elitism, mutate_after_crossover=True, parallelism=False, threads_per_calculation=3, mutation_rate=2)
    result.save('best_Al10', directory='')
    return result, strategies


def test_Al10_mp():
    molecule = Molecule.load('al_n.inp', rand_range=5)
    def load():
        return molecule
    mutate_methods = Mutate([swap_mutate, mutate_angles, mutate_distances], [1, 1, 1])
    crossover_methods = Crossover([crossover_1, crossover_2, crossover_n], [1, 1, 1])
    create_methods = Create([randomize, mutate_first], [1, 1])
    load_first_parent = load
    strategies = Strategies([create_methods, mutate_methods, crossover_methods], [0, 1, 1])
    max_age = 20
    pool_size = 16
    max_seconds = None
    time_tolerance = None
    crossover_elitism = None
    fitness_param = '!RKS STATE 1.1 Energy'

    result, strategies = optimize(molecule, fitness_param, strategies, max_age, pool_size, max_seconds, time_tolerance,
    crossover_elitism, mutate_after_crossover=False, parallelism=True, elit_size=None, elitism_rate=None,
    generations_tolerance=None, threads_per_calculation=1, max_gens=None, mutation_rate=2)
    result.save('best_Al10_mp', directory='')
    return result, strategies


class Genetic:
    def __init__(self, first_molecule, strategies, max_age, pool_size, fitness_param, mutate_after_crossover, 
        elitism_rate, crossover_elitism, freedom_rate, mutation_rate, parallelism, threads_per_calc, max_seconds, 
        max_gens, time_tolerance, generations_tolerance):
        self.first_molecule = first_molecule
        self.strategies = strategies
        self.max_age = max_age
        self.pool_size = pool_size
        self.fitness_param = fitness_param
        self.mutate_after_crossover = mutate_after_crossover
        self.elitism_rate = elitism_rate
        self.crossover_elitism = crossover_elitism
        self.freedom_rate = freedom_rate
        self.mutation_rate = mutation_rate
        self.parallelism = parallelism
        self.threads_per_calc = threads_per_calc
        self.max_seconds = max_seconds
        self.max_seconds = max_gens
        self.time_tolerance = time_tolerance
        self.generations_tolerance = generations_tolerance
        self.optg_threads = threads_per_calc if not parallelism else threads_per_calc * pool_size
    
    def fn_get_fitness(self, molecule):
        if molecule.was_optg:
            return - molecule.output_values[self.fitness_param]
        return - float(molecule.get_value([self.fitness_param], nthreads=self.threads_per_calc)[self.fitness_param])

    def fn_local_optimization(self, molecule):
        return optg(molecule, self.fitness_param, nthreads = self.optg_threads)

    def fn_catch(self, candidate):
        os.remove(f'data/{candidate.label}.inp')
        os.remove(f'data/{candidate.label}.out')
        os.remove(f'data/{candidate.label}.xml')



if __name__ == '__main__':
    test_Al10_mp()
