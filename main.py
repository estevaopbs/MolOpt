import os
from genetic import *
from molecular import *


class Molecular_improvement(Genetic):
    """[summary]

    :param Genetic: [description]
    :type Genetic: [type]
    """    
    def __init__(self, first_genes, fitness_param, strategies, max_age, pool_size, mutate_after_crossover, 
        crossover_elitism, elitism_rate, freedom_rate, parallelism, local_opt, max_seconds, time_toler,
        gens_toler, max_gens, threads_per_calc, save_directory):
        """[summary]

        :param first_genes: [description]
        :type first_genes: [type]
        :param fitness_param: [description]
        :type fitness_param: [type]
        :param strategies: [description]
        :type strategies: [type]
        :param max_age: [description]
        :type max_age: [type]
        :param pool_size: [description]
        :type pool_size: [type]
        :param mutate_after_crossover: [description]
        :type mutate_after_crossover: [type]
        :param crossover_elitism: [description]
        :type crossover_elitism: [type]
        :param elitism_rate: [description]
        :type elitism_rate: [type]
        :param freedom_rate: [description]
        :type freedom_rate: [type]
        :param parallelism: [description]
        :type parallelism: [type]
        :param local_opt: [description]
        :type local_opt: [type]
        :param max_seconds: [description]
        :type max_seconds: [type]
        :param time_toler: [description]
        :type time_toler: [type]
        :param gens_toler: [description]
        :type gens_toler: [type]
        :param max_gens: [description]
        :type max_gens: [type]
        :param threads_per_calc: [description]
        :type threads_per_calc: [type]
        :param save_directory: [description]
        :type save_directory: [type]
        """        
        super().__init__(first_genes, fitness_param, strategies, max_age, pool_size, mutate_after_crossover, 
        crossover_elitism, elitism_rate, freedom_rate, parallelism, local_opt, max_seconds, time_toler,
        gens_toler, max_gens, save_directory)
        self.threads_per_calc = threads_per_calc

    def get_fitness(self, candidate):
        """[summary]

        :param candidate: [description]
        :type candidate: [type]
        :return: [description]
        :rtype: [type]
        """        
        molecule = candidate.genes
        file_name = candidate.label
        if candidate.label == '0_0':
            return - float(molecule.get_value([self.fitness_param], document=file_name, 
                directory=self.save_directory + '/data', 
                nthreads=self.threads_per_calc * self.pool_size)[self.fitness_param])
        if molecule.was_optg:
            return - molecule.output_values[self.fitness_param]
        return - float(molecule.get_value([self.fitness_param], document=file_name, 
            directory=self.save_directory + '/data', 
            nthreads=self.threads_per_calc)[self.fitness_param])

    @staticmethod
    def swap_mutate(parent):
        """[summary]

        :param parent: [description]
        :type parent: [type]
        :return: [description]
        :rtype: [type]
        """        
        return swap_mutate(parent.genes)

    @staticmethod
    def mutate_angles(parent):
        """[summary]

        :param parent: [description]
        :type parent: [type]
        :return: [description]
        :rtype: [type]
        """        
        return mutate_angles(parent.genes)

    @staticmethod
    def mutate_distances(parent):
        """[summary]

        :param parent: [description]
        :type parent: [type]
        :return: [description]
        :rtype: [type]
        """        
        return mutate_distances(parent.genes)

    @staticmethod
    def crossover_1(parent, donor):
        """[summary]

        :param parent: [description]
        :type parent: [type]
        :param donor: [description]
        :type donor: [type]
        :return: [description]
        :rtype: [type]
        """        
        return crossover_1(parent.genes, donor.genes)

    @staticmethod
    def crossover_2(parent, donor):
        """[summary]

        :param parent: [description]
        :type parent: [type]
        :param donor: [description]
        :type donor: [type]
        :return: [description]
        :rtype: [type]
        """        
        return crossover_2(parent.genes, donor.genes)

    @staticmethod
    def crossover_n(parent, donor):
        """[summary]

        :param parent: [description]
        :type parent: [type]
        :param donor: [description]
        :type donor: [type]
        :return: [description]
        :rtype: [type]
        """        
        return crossover_n(parent.genes, donor.genes)

    @staticmethod
    def randomize(parent):
        """[summary]

        :param parent: [description]
        :type parent: [type]
        :return: [description]
        :rtype: [type]
        """        
        return randomize(parent.genes)

    def local_optimize(self, molecule):
        """[summary]

        :param molecule: [description]
        :type molecule: [type]
        :return: [description]
        :rtype: [type]
        """        
        return optg(molecule, self.fitness_param, nthreads = self.pool_size * self.threads_per_calc)

    @staticmethod
    def catch(candidate):
        """[summary]

        :param candidate: [description]
        :type candidate: [type]
        """        
        os.remove(f'data/{candidate.label}.inp')
        os.remove(f'data/{candidate.label}.out')
        os.remove(f'data/{candidate.label}.xml')

    @staticmethod
    def save(candidate, file_name, directory):
        """[summary]

        :param candidate: [description]
        :type candidate: [type]
        :param file_name: [description]
        :type file_name: [type]
        :param directory: [description]
        :type directory: [type]
        """        
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
    create_methods = Create([Molecular_improvement.randomize, mutate_first, mutate_best], [1, 0.5, 0.5])
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
