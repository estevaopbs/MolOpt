import os
from genetic import Genetic, Create, Mutate, Crossover, Strategies
from molecular import *


class Molecular_improvement(Genetic):
    def get_fitness(self, molecule):
        if molecule.was_optg:
            return - molecule.output_values[self.fitness_param]
        return - float(molecule.get_value([self.fitness_param], nthreads=self.threads_per_calc)[self.fitness_param])

    def flocal_optimizate(self, molecule):
        return optg(molecule, self.fitness_param, nthreads = self.pool_size * self.threads_per_calc)

    def catch(self, candidate):
        os.remove(f'data/{candidate.label}.inp')
        os.remove(f'data/{candidate.label}.out')
        os.remove(f'data/{candidate.label}.xml')


if __name__ == '__main__':
    mutate_methods = Mutate([swap_mutate, mutate_angles, mutate_distances], [1, 1, 1])
    crossover_methods = Crossover([crossover_1, crossover_2, crossover_n], [1, 1, 1])
    create_methods = Create([randomize, Genetic.mutate_first], [1, 1])
    strategies = Strategies([mutate_methods, crossover_methods, create_methods], [1, 1, 0])
    Al10_test = Molecular_improvement(
        first_genes = Molecule.load('al_n.inp', (0, 3)),
        fitness_param = '!RKS STATE 1.1 Energy',
        strategies = strategies,
        max_age = 200,
        pool_size = 8,
        mutate_after_crossover = False,
        crossover_elitism = None,
        elitism_rate = None,
        freedom_rate = 3,
        parallelism = True,
        threads_per_calc=2,
        max_seconds = None,
        time_toler=None,
        gens_toler = None,
        max_gens = None
    )
    Al10_test.run()