import os
from genetic import *
from molecular import *


class Molecular_improvement(Genetic):
    def __init__(self, first_genes, fitness_param, strategies, max_age, pool_size, mutate_after_crossover, 
        crossover_elitism, elitism_rate, freedom_rate, parallelism, max_seconds, time_toler,
        gens_toler, max_gens, threads_per_calc):
        super().__init__(first_genes, fitness_param, strategies, max_age, pool_size, mutate_after_crossover, 
        crossover_elitism, elitism_rate, freedom_rate, parallelism, max_seconds, time_toler,
        gens_toler, max_gens)
        self.threads_per_calc = threads_per_calc

    def get_fitness(self, candidate):
        molecule = candidate.genes
        file_name = candidate.label
        if candidate.label == '0_0':
            return - float(molecule.get_value([self.fitness_param], document=file_name, 
                nthreads=self.threads_per_calc * self.pool_size)[self.fitness_param])
        if molecule.was_optg:
            return - molecule.output_values[self.fitness_param]
        return - float(molecule.get_value([self.fitness_param], document=file_name, 
            nthreads=self.threads_per_calc)[self.fitness_param])

    #def local_optimize(self, molecule):
    #    return optg(molecule, self.fitness_param, nthreads = self.pool_size * self.threads_per_calc)

    @staticmethod
    def catch(candidate):
        os.remove(f'data/{candidate.label}.inp')
        os.remove(f'data/{candidate.label}.out')
        os.remove(f'data/{candidate.label}.xml')

    def save(candidate, file_name, directory):
        candidate.genes.save(file_name, directory)


if __name__ == '__main__':
    mutate_methods = Mutate([swap_mutate, mutate_angles, mutate_distances], [1, 1, 1])
    crossover_methods = Crossover([crossover_1, crossover_2, crossover_n], [1, 1, 1])
    create_methods = Create([randomize, mutate_first], [1, 1])
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