from molopt import *


if __name__ == '__main__':
    mutate_methods = Mutate([MolOpt.swap_mutate, MolOpt.mutate_angles, MolOpt.mutate_distances], [1, 1, 1])
    crossover_methods = Crossover([MolOpt.crossover_1, MolOpt.crossover_2, MolOpt.crossover_n], [1, 1, 1])
    create_methods = Create([MolOpt.randomize, mutate_first, mutate_best], [1, 0.5, 0.5])
    strategies = Strategies([mutate_methods, crossover_methods, create_methods], [1, 1, 0])
    Al10_test = MolOpt(
        first_genes = Molecule.load('al_n.inp', (0, 3)),
        fitness_param = '!RKS STATE 1.1 Energy',
        strategies = strategies,
        max_age = 200,
        pool_size = 3,
        mutate_after_crossover = False,
        crossover_elitism = None,
        elitism_rate = None,
        freedom_rate = 2,
        parallelism = True,
        local_opt = False,
        threads_per_calc = 1,
        max_seconds = None,
        time_toler = None,
        gens_toler = None,
        max_gens = None,
        save_directory = None
    )
    Al10_test.run()