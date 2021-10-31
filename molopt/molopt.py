import os
from molopt.genetic import *
from molopt.molecular import *


class MolOpt(Genetic):
    """[summary]

    :param Genetic: [description]
    :type Genetic: [type]
    """    
    def __init__(self, first_genes: Molecule, fitness_param: str, strategies: Strategies, max_age: int, pool_size: int, 
        mutate_after_crossover: bool, crossover_elitism: list[int], elitism_rate: list[int], freedom_rate: int, 
        parallelism: bool, local_opt: bool, max_seconds: int | float | None, time_toler: int | float | None, 
        gens_toler: int | None, max_gens: int | None, save_directory: str, threads_per_calc: int):
        """[summary]

        :param first_genes: [description]
        :type first_genes: Molecule
        :param fitness_param: [description]
        :type fitness_param: str
        :param strategies: [description]
        :type strategies: Strategies
        :param max_age: [description]
        :type max_age: int
        :param pool_size: [description]
        :type pool_size: int
        :param mutate_after_crossover: [description]
        :type mutate_after_crossover: bool
        :param crossover_elitism: [description]
        :type crossover_elitism: list[int]
        :param elitism_rate: [description]
        :type elitism_rate: list[int]
        :param freedom_rate: [description]
        :type freedom_rate: int
        :param parallelism: [description]
        :type parallelism: bool
        :param local_opt: [description]
        :type local_opt: bool
        :param max_seconds: [description]
        :type max_seconds: int | float | None
        :param time_toler: [description]
        :type time_toler: int | float | None
        :param gens_toler: int | None
        :type gens_toler: int
        :param max_gens: [description]
        :type max_gens: int | None
        :param save_directory: [description]
        :type save_directory: str
        :param threads_per_calc: [description]
        :type threads_per_calc: int
        """        
        super().__init__(first_genes, fitness_param, strategies, max_age, pool_size, mutate_after_crossover, 
        crossover_elitism, elitism_rate, freedom_rate, parallelism, local_opt, max_seconds, time_toler,
        gens_toler, max_gens, save_directory)
        self.threads_per_calc = threads_per_calc

    def get_fitness(self, candidate: Chromosome) -> float:
        """[summary]

        :param candidate: [description]
        :type candidate: Chromosome
        :return: [description]
        :rtype: float
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
    def swap_mutate(parent: Chromosome) -> Molecule:
        """[summary]

        :param parent: [description]
        :type parent: Chromosome
        :return: [description]
        :rtype: Molecule
        """        
        return swap_mutate(parent.genes)

    @staticmethod
    def mutate_angles(parent: Chromosome) -> Molecule:
        """[summary]

        :param parent: [description]
        :type parent: Chromosome
        :return: [description]
        :rtype: Molecule
        """        
        return mutate_angles(parent.genes)

    @staticmethod
    def mutate_distances(parent: Chromosome) -> Molecule:
        """[summary]

        :param parent: [description]
        :type parent: Chromosome
        :return: [description]
        :rtype: Molecule
        """        
        return mutate_distances(parent.genes)

    @staticmethod
    def crossover_1(parent: Chromosome, donor: Chromosome) -> Molecule:
        """[summary]

        :param parent: [description]
        :type parent: Chromosome
        :param donor: [description]
        :type donor: Chromosome
        :return: [description]
        :rtype: Molecule
        """        
        return crossover_1(parent.genes, donor.genes)

    @staticmethod
    def crossover_2(parent: Chromosome, donor: Chromosome) -> Molecule:
        """[summary]

        :param parent: [description]
        :type parent: Chromosome
        :param donor: [description]
        :type donor: Chromosome
        :return: [description]
        :rtype: Molecule
        """             
        return crossover_2(parent.genes, donor.genes)

    @staticmethod
    def crossover_n(parent: Chromosome, donor: Chromosome) -> Molecule:
        """[summary]

        :param parent: [description]
        :type parent: Chromosome
        :param donor: [description]
        :type donor: Chromosome
        :return: [description]
        :rtype: Molecule
        """           
        return crossover_n(parent.genes, donor.genes)

    @staticmethod
    def randomize(parent: Chromosome) -> Molecule:
        """[summary]

        :param parent: [description]
        :type parent: Chromosome
        :return: [description]
        :rtype: Molecule
        """              
        return randomize(parent.genes)

    def local_optimize(self, candidate: Chromosome) -> Molecule:
        """[summary]

        :param candidate: [description]
        :type candidate: Chromosome
        :return: [description]
        :rtype: Molecule
        """              
        return optg(candidate.genes, self.fitness_param, nthreads = self.pool_size * self.threads_per_calc)

    @staticmethod
    def catch(candidate: Chromosome) -> None:
        """[summary]

        :param candidate: [description]
        :type candidate: Chromosome
        """              
        os.remove(f'data/{candidate.label}.inp')
        os.remove(f'data/{candidate.label}.out')
        os.remove(f'data/{candidate.label}.xml')

    @staticmethod
    def save(candidate: Chromosome, file_name: str, directory: str) -> None:
        """[summary]

        :param candidate: [description]
        :type candidate: Chromosome
        :param file_name: [description]
        :type file_name: str
        :param directory: [description]
        :type directory: str
        """            
        candidate.genes.save(file_name, directory)
