from genetic import *


def test_10Nb():
    molecule = Molecule.load('nb_n_molpro.inp', rand_range=5)
    mutate_methods = Mutate([Mutate.swap_mutate, Mutate.mutate_angles, Mutate.mutate_distances], [1, 1, 1])
    crossover_methods = Crossover([Crossover.crossover_n, Crossover.crossover_1, Crossover.crossover_2], [1, 1, 1])
    create_methods = Create([Create.randomize, Create.mutate_first], [1, 1])
    strategies = Strategies([create_methods, mutate_methods, crossover_methods], [0, 1, 1])
    max_age = 5
    pool_size = 20
    #elit_size = 0.1
    #elitism_rate = 0.1
    max_seconds = 100800
    time_tolerance = 30
    crossover_elitism = lambda x: 1
    fitness_param = '!RKS STATE 1.1 Energy'

    optimize(molecule, fitness_param, strategies, max_age, pool_size, max_seconds, time_tolerance, crossover_elitism,
    True)


if __name__ == '__main__':
    test_10Nb()
