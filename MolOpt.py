import os
from __future__ import annotations
from MolOpt.genetic import *
from MolOpt.molecular import *


"""Atomic clusters geometry optimization using genetic algorithm with Molpro
"""


Fitness = float


class MolOpt(Genetic):
    """Molecular geometry optimization class

    :param Genetic: Genetic algorithm abstract class
    :type Genetic: ABC
    """
    def __init__(self, first_molecule: Molecule, energy_param: str, strategies: Strategies, max_age: int | None, 
        pool_size: int, mutate_after_crossover: bool, crossover_elitism: list[int], elitism_rate: list[int], 
        freedom_rate: int, parallelism: bool, local_opt: bool, max_seconds: numeric | None, 
        time_toler: numeric | None, gens_toler: int | None, max_gens: int | None, 
        save_directory: str, threads_per_calc: int) -> MolOpt:
        """Initializes the MolOpt object

        :param first_molecule: Molecule which is wanted to be optimized
        :type first_molecule: Molecule
        :param energy_param: The string which precedes the energy value in Molpro's output
        :type energy_param: str
        :param strategies: Strategies object
        :type strategies: Strategies
        :param max_age: The max amount of times a Chromosome can suffer chaging strategies without improve its fitness.
            if it's None then only improvements will be accepted
        :type max_age: int | None
        :param pool_size: The amount of candidates being optimized together
        :type pool_size: int
        :param mutate_after_crossover: If it's True then, after each crossover operation, the resultant child Chromosome
            will suffer a mutate operation before return to the genetic algorithm, but if it's false the mutation 
            doesn't occur and the child Chromosome will be returned immediately after the crossover
        :type mutate_after_crossover: bool
        :param crossover_elitism: The rate each candidate tends to be selected to be the gene's donor in any crossover 
            operation from the best to the worst. Its lenght must be equal pool_size value. If pool_size is 3 and 
            crossover_elitism is [3, 2, 1] the best candidate has the triple of the chance to be selected than the 
            worst, the medium candidate has double. It can also receive None, and it means all candidates would be 
            equally probable to be selected for being the genes' donor on a crossover
        :type crossover_elitism: list[numeric] | None
        :param elitism_rate: Reprodution rate of each candidate, from the best to the worst. the sum of its elements 
            also must be less or equal than pool_size. If pool_size is 16 and elitism_rate is [4, 3, 2] it means the 
            best candidate in the current generation's pool of candidates will provide 4 descendants for the next 
            generation, the second best will provide 3 and the third best will provide two, then the remains 7 available
            spaces in the next generation's pool will be filled with one descendant of each one of the seven next 
            candidates in this order
        :type elitism_rate: list[int] | None
        :param freedom_rate: The number of candidate generation strategies (Mutate, Crossover and Create) the candidate 
            will suffer aways a new candidate is needed to be generated
        :type freedom_rate: int
        :param parallelism: If it's True then each fitness calculation will be done in a different process, what changes
            the whole dynamics of the genetic algorithm. With paraellism enabled, the concept of generations emerges as
            we can have different candidates being caculated at the same time. If it's False, there will be no
            generations and candidates' fitnesses will be calculated sequentially
        :type parallelism: bool
        :param local_opt: If its True makes that every time the algorithm genetic gets a new best candidate it is sent 
            to the local_optimize function (which in this case must be override by the subclass) that is supposed to
            perform a local optimization (in the solution-space of the specific problem) over the genes of the best 
            candidate, and then return a new best candidate with the optimized genes
        :type local_opt: bool
        :param max_seconds: The max amount of seconds the genetic algorithm can run. Once exceeded this amount, the
            the running will be stoped and the best candidate will be returned. It can also receive None, and in this 
            case this limit wouldn't exist
        :type max_seconds: numeric | None
        :param time_toler: The max amount of seconds the algorithm can still running without has any improvements on its
            best candidate's fitness. Once exceeded this amount, the running will be stoped and the best candidate will 
            be returned. It can also receive None, and in this case this limit wouldn't exist
        :type time_toler: numeric | None
        :param gens_toler: The maximum amount of generations the algorithm genetic can run in sequence without having
            any improvement on it's best parent fitness. It can also receive None and in this case this limit wouldn't 
            exist. It only works when parallelism is True, otherwise it doesn't affect anything
        :type gens_toler: numeric | None
        :param max_gens: The max amount of generations the genetic algorithm can run. Once exceeded this amount, the
            the running will be stoped and the best candidate will be returned. It can also receive None, and in this 
            case this limit wouldn't exist. It only works when parallelism is True, otherwise it doesn't affect anything
        :type max_gens: numeric | None
        :param save_directory: The directory address relative to __main__ where the outputs will be saved. If its None
            than it will receive the instant of time the running started
        :type save_directory: str
        :param threads_per_calc: Number of threads useds in each Molpro calculation
        :type threads_per_calc: int
        """
        self.energy_param = energy_param
        self.threads_per_calc = threads_per_calc
        super().__init__(first_molecule, strategies, max_age, pool_size, mutate_after_crossover, crossover_elitism, 
            elitism_rate, freedom_rate, parallelism, local_opt, max_seconds, time_toler,gens_toler, max_gens, 
            save_directory)

    def get_fitness(self, candidate: Chromosome) -> float:
        """Receives a candidate's Chromosome and returns its fitness

        :param candidate: Candidate which fitness must be calculated
        :type candidate: Chromosome
        :return: Candidate's fitness
        :rtype: float
        """
        molecule = candidate.genes
        file_name = candidate.label
        if self.energy_param in molecule.output_values.keys():
            return - molecule.output_values[self.energy_param]
        if candidate.label == '0_0' and self.parallelism:
            return - (molecule.get_value([self.energy_param], document=file_name, 
                directory=self.save_directory + '/data', 
                nthreads=self.threads_per_calc * self.pool_size)[self.energy_param])
        if molecule.was_optg:
            return - molecule.output_values[self.energy_param]
        return - (molecule.get_value([self.energy_param], document=file_name, 
            directory=self.save_directory + '/data', 
            nthreads=self.threads_per_calc)[self.energy_param])

    @staticmethod
    def swap_mutate(parent: Chromosome) -> Molecule:
        """Returns a parent's molecule's copy with randomly swapped places parameters

        :param parent: Candidate which Molecule will suffer swap_mutate
        :type parent: Chromosome
        :return: New Molecule
        :rtype: Molecule
        """
        return swap_mutate(parent.genes)

    @staticmethod
    def mutate_angles(parent: Chromosome) -> Molecule:
        """Returns parent's molecule's copy with some random angle parameter randomized between 0 and 360 degrees

        :param parent: Candidate which Molecule will suffer mutate_angles
        :type parent: Chromosome
        :return: New Molecule
        :rtype: Molecule
        """
        return mutate_angles(parent.genes)

    @staticmethod
    def mutate_distances(parent: Chromosome) -> Molecule:
        """Returns a parent's molecule's copy with some random distance parameter randomized in the range gave by 
        parent.genes.rand_range

        :param parent: Candidate which Molecule will suffer mutate_distances
        :type parent: Chromosome
        :return: New Molecule
        :rtype: Molecule
        """
        return mutate_distances(parent.genes)

    @staticmethod
    def atom_displacement_mutate(parent: Chromosome) -> Molecule:
        """Returns a parent's molecule's copy Selects two random atoms then swaps its positions

        :param parent: Candidate which Molecule will suffer atom_displacement_mutate
        :type parent: Chromosome
        :return: New Molecule
        :rtype: Molecule
        """
        return atom_displacement_mutate(parent.genes)

    @staticmethod
    def crossover_1(parent: Chromosome, donor: Chromosome) -> Molecule:
        """Produces a new molecule with the crossover of parent's and donor's molecules by cutting each one in one point
        and combining the resultant pieces.

        :param parent: Candidate which Molecule will suffer crossover_1
        :type parent: Chromosome
        :param donor: Candidate which will donate parameters for the crossover_1 operation
        :type donor: Chromosome
        :return: Child Molecule
        :rtype: Molecule
        """
        return crossover_1(parent.genes, donor.genes)

    @staticmethod
    def crossover_2(parent: Chromosome, donor: Chromosome) -> Molecule:
        """Produces a new molecule with the crossover of parent's and donor's molecules by cutting each one in two 
        points and combining the resultant pieces

        :param parent: Candidate which Molecule will suffer crossover_1
        :type parent: Chromosome
        :param donor: Candidate which will donate parameters for the crossover_1 operation
        :type donor: Chromosome
        :return: Child Molecule
        :rtype: Molecule
        """
        return crossover_2(parent.genes, donor.genes)

    @staticmethod
    def crossover_n(parent: Chromosome, donor: Chromosome) -> Molecule:
        """Returns a new molecule which randomly carries parameters from the parent's and donor's molecules

        :param parent: Candidate which Molecule will suffer crossover_1
        :type parent: Chromosome
        :param donor: Candidate which will donate parameters for the crossover_1 operation
        :type donor: Chromosome
        :return: Child Molecule
        :rtype: Molecule
        """
        return crossover_n(parent.genes, donor.genes)

    @staticmethod
    def randomize(parent: Chromosome) -> Molecule:
        """Returns a parent's molecule's copy with all distances and angles parameters randomized

        :param parent: Candidate which Molecule will suffer randomize operation
        :type parent: Chromosome
        :return: New Molecule
        :rtype: Molecule
        """
        return randomize(parent.genes)

    @staticmethod
    def ieq_mutate(parent: Chromosome) -> Molecule:
        """Returns a parent's molecule's copy with two random parameters of the same kind (either distance or angle)
        equalized 

        :param parent: Candidate which Molecule will suffer ieq_mutate operation
        :type parent: Chromosome
        :return: New Molecule
        :rtype: Molecule
        """        
        return ieq_mutate(parent.genes)

    @staticmethod
    def atom_swap_mutate(parent: Chromosome) -> Molecule:
        """Returns a parent's molecule's copy with two atoms positions' swapped

        :param parent: Candidate which Molecule will suffer atom_swap_mutate operation
        :type parent: Chromosome
        :return: New Molecule
        :rtype: Molecule
        """        
        return atom_swap_mutate(parent.genes)

    def local_optimize(self, candidate: Chromosome) -> Molecule:
        """Executes geometric optimization over the candidate's molecule using Molpro and returns a new molecule with 
        the optimized geometry

        :param candidate: Candidate which Molecule will suffer local_optimize operation
        :type candidate: Chromosome
        :return: Optimized Molecule
        :rtype: Molecule
        """
        return optg(candidate.genes, self.energy_param, nthreads = self.pool_size * self.threads_per_calc)

    def catch(self, candidate: Chromosome) -> None:
        """Static method which will be executed if an error occurs during a candidate generation

        :param candidate: Candidate which during generation some exception occurred
        :type candidate: Chromosome
        """
        os.remove(f'{self.save_directory}/data/{candidate.label}.inp')
        os.remove(f'{self.save_directory}/data/{candidate.label}.out')
        os.remove(f'{self.save_directory}/data/{candidate.label}.xml')

    @staticmethod
    def save(candidate: Chromosome, file_name: str, directory: str) -> None:        
        """Saves the candidate data in a .inp document

        :param candidate: Candidate which data will be saved
        :type candidate: Chromosome
        :param file_name: Document's name
        :type file_name: str
        :param directory: Directory where the document will be saved
        :type directory: str
        """
        candidate.genes.save(file_name, directory)
