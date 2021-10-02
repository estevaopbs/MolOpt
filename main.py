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
    crossover_elitism = lambda x: 1
    fitness_param = '!RKS STATE 1.1 Energy'

    result, strategies = optimize(molecule, fitness_param, strategies, max_age, pool_size, max_seconds, time_tolerance, 
    crossover_elitism, mutate_after_crossover=True, parallelism=False, threads_per_calculation=3, mutation_rate=2)
    result.save('best_Al10', directory='')
    return result


def test_Al10_mp():  # não funciona e não sei porque
    molecule = Molecule.load('al_n.inp', rand_range=5)
    mutate_methods = Mutate([Mutate.swap_mutate, Mutate.mutate_angles, Mutate.mutate_distances], [1, 1, 1])
    crossover_methods = Crossover([Crossover.crossover_n, Crossover.crossover_1, Crossover.crossover_2], [1, 1, 1])
    create_methods = Create([Create.randomize, Create.mutate_first], [1, 1])
    strategies = Strategies([create_methods, mutate_methods, crossover_methods], [0, 1, 1])
    max_age = 10
    pool_size = 3
    max_seconds = 3600 * 10
    time_tolerance = 3600
    crossover_elitism = lambda x: 1
    fitness_param = '!RKS STATE 1.1 Energy'

    result, strategies = optimize(molecule, fitness_param, strategies, max_age, pool_size, max_seconds, time_tolerance,
    crossover_elitism, mutate_after_crossover=True, parallelism=True, elit_size=1, elitism_rate=[2],
    generations_tolerance=20, threads_per_calculation=1, max_gens=500, mutation_rate=2)
    result.save('best_Al10_mp', directory='')
    return result


if __name__ == '__main__':
    test_Al10_mp()
