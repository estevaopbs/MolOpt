import os
from genetic import *
from molecular import *


class Molecular_improvement(Genetic):
    def __init__(self, first_genes, fitness_param, strategies, max_age, pool_size, mutate_after_crossover, 
        crossover_elitism, elitism_rate, freedom_rate, parallelism, local_opt, max_seconds, time_toler,
        gens_toler, max_gens, threads_per_calc, save_directory):
        super().__init__(first_genes, fitness_param, strategies, max_age, pool_size, mutate_after_crossover, 
        crossover_elitism, elitism_rate, freedom_rate, parallelism, local_opt, max_seconds, time_toler,
        gens_toler, max_gens, save_directory)
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

    @staticmethod
    def swap_mutate(parent): return swap_mutate(parent.genes)

    @staticmethod
    def mutate_angles(parent): return mutate_angles(parent.genes)

    @staticmethod
    def mutate_distances(parent): return mutate_distances(parent.genes)

    @staticmethod
    def crossover_1(parent, donor): return crossover_1(parent.genes, donor.genes)

    @staticmethod
    def crossover_2(parent, donor): return crossover_2(parent.genes, donor.genes)

    @staticmethod
    def crossover_n(parent, donor): return crossover_n(parent.genes, donor.genes)

    @staticmethod
    def randomize(parent):
        return randomize(parent.genes)

    def local_optimize(self, molecule):
        return optg(molecule, self.fitness_param, nthreads = self.pool_size * self.threads_per_calc)

    @staticmethod
    def catch(candidate):
        os.remove(f'data/{candidate.label}.inp')
        os.remove(f'data/{candidate.label}.out')
        os.remove(f'data/{candidate.label}.xml')

    @staticmethod
    def save(candidate, file_name, directory):
        candidate.genes.save(file_name, directory)


if __name__ == '__main__':
    mutate_methods = Mutate(
        [
            Molecular_improvement.swap_mutate, 
            Molecular_improvement.mutate_angles, 
            Molecular_improvement.mutate_distances], 
            [1, 1, 1]
        )
    crossover_methods = Crossover(
        [
            Molecular_improvement.crossover_1, 
            Molecular_improvement.crossover_2, 
            Molecular_improvement.crossover_n], 
            [1, 1, 1]
            )
    create_methods = Create([Molecular_improvement.randomize, mutate_first], [1, 1])
    strategies = Strategies([mutate_methods, crossover_methods, create_methods], [1, 1, 0])
    Al10_test = Molecular_improvement(
        first_genes = Molecule.load('al_n.inp', (0, 3)),
        fitness_param = '!RKS STATE 1.1 Energy',
        strategies = strategies,
        max_age = 20,
        pool_size = 3,
        mutate_after_crossover = False,
        crossover_elitism = None,
        elitism_rate = None,
        freedom_rate = 3,
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