from __future__ import annotations
from math import sqrt, floor
import os
import random
from typing import Tuple
from MolOpt.genetic.genetic import Genetic, Strategies, Chromosome, Create, Mutate, Crossover
from MolOpt.molecular import particle_permutation, piece_displacement, piece_reflection
from MolOpt.molecular import randomize, piece_rotation, enlarge_reduce
import numpy as np
from MolOpt.Molpro_Molecule import Molpro_Molecule


"""Molecular geometry optimization package using genetic algorithm and Molpro
"""


class MolOpt(Genetic):
    """Molecular geometry optimization class

    Parameters
    ----------
    Genetic : ABC
        Genetic algorithm abstract class
    """
    __slots__ = ('threads_per_calc')

    def __init__(self, first_molecule: Molpro_Molecule, displacement_range: Tuple[float, float],
        rotation_range: Tuple[float, float], piece_size_range: Tuple[int, int], distance_range: Tuple[float, float],
        enlarge_reduce_range: Tuple[int, int], strategies: Strategies, max_age: int | None, pool_size: int,
        mutate_after_crossover: bool, crossover_elitism: list[float] | None, elitism_rate: list[int] | None,
        freedom_rate: int, parallelism: bool, aways_local_opt: bool, time_toler_opt: float | None,
        gens_toler_opt: int | None, max_seconds: float | None, time_toler: float | None, gens_toler: int | None,
        max_gens: int | None, max_fitness: float | None, save_directory: str | None,
        adapting_args: Tuple[str, str] | None, adapting_rate: float | None, threads_per_calc: int) -> None:
        """Initializes the MolOpt object

        Parameters
        ----------
        first_molecule : Molpro_Molecule
            Molecule which is wanted to be optimized
        displacement_range : Tuple[float, float]
            Range in which particles can be randomly displaced
        rotation_range : Tuple[float, float]
            Range in which particles can be randomly rotated
        piece_size_range : Tuple[int, int]
            Range in which pieces will be picked
        distance_range : Tuple[float, float]
            Range in which random distances will be generated when randomize function is called
        enlarge_reduce_range : Tuple[int, int]
            Range of the rate in which internal distances will be multiplied to enlarge or reduce the molecule 
        strategies : Strategies
            Set of strategies and its rates
        max_age : int | None
            The max amount of times a Chromosome can suffer chaging strategies without improve its fitness
        pool_size : int
            The amount of candidates being optimized together
        mutate_after_crossover : bool
            If its True, then after after each crossover operation, the resultant child Chromosome will suffer a mutate
            operation
        crossover_elitism : list[float] | None
            The rates each candidate tends to be selected to be the gene's donor in any crossover operation from the
            best to the worst
        elitism_rate : list[int] | None
            Reprodution rates of each candidate, from the best to the worst
        freedom_rate : int
            The number of candidate generation strategies (Mutate, Crossover and Create) the candidate will suffer aways
            a new candidate is needed to be generated
        parallelism : bool
            If it's True then each fitness calculation will be done in a different process
        aways_local_opt : bool
            If it's True then every time the algorithm genetic gets a new best candidate it is sent to the
            local_optimize function
        time_toler_opt : float | None
            If the time between two improvements is bigger than this, then the new improvement will be sent to the
            local_optimization function
        gens_toler_opt : int | None
            If the generations that passed between two improvements are more than this, then the new improvement will be
            sent to the local_optimization function
        max_seconds : float | None
            The maximum amount of seconds the genetic algorithm can run
        time_toler : float | None
            The maximum amount of seconds the algorithm can still running without having any improvements on its best
            candidate's fitness
        gens_toler : int | None
            The maximum amount of generations the algorithm genetic can run in sequence without having any improvement
            on it's best candidate's fitness
        max_gens : int | None
            The maximum amount of generations the genetic algorithm can run
        max_fitness : float | None
            The maximum fitness the genetic algorithm can achieve
        save_directory : str | None
            The directory address relative to __main__ where the outputs will be saved
        adapting_rate : float | None
            The rate the strategies and its methods rates will be adapted to the historical rate
        adapting_args : Tuple[str, str] | None
            Arguments that define the adapting function. It must have two strings, one to define the adapting chance and
            one to define the adapting target. The adapting chance possible arguments are "frequency" which increases
            the chance to adapt the lower is the frequency the improvements are achieved and "fitness_difference" which
            increases the chance to adapt the less significance the improvements have. The adapting target possible
            arguments are "fitness_increasement" which privilegies the strategies that most contributed with the fitness
            increasement, "improvements" that privilegies the strategies that generated most best candidates and
            "lineage" that privilegies the strategies that generated more ancestors
        threads_per_calc : int
            Number of threads useds in each Molpro calculation
        """
        self.threads_per_calc = threads_per_calc
        super().__init__(first_molecule, strategies, max_age, pool_size, mutate_after_crossover, crossover_elitism, 
            elitism_rate, freedom_rate, parallelism, aways_local_opt, time_toler_opt, gens_toler_opt, max_seconds,
            time_toler,gens_toler, max_gens, max_fitness, save_directory, adapting_rate, adapting_args,
            {'displacement_range': displacement_range, 'rotation_range': rotation_range,
            'piece_size_range': piece_size_range, 'distance_range': distance_range,
            'enlarge_reduce_range': enlarge_reduce_range})

    def run(self) -> Chromosome:
        """Starts the genetic algorithm execution

        Returns
        -------
        Chromosome
            Best candidate
        """
        best = super().run()
        os.rmdir(f'{self.save_directory}/data')
        return best

    def get_fitness(self, candidate: Chromosome) -> float:
        """Receives a candidate's Chromosome and returns its fitness

        Parameters
        ----------
        candidate : Chromosome
            Candidate which fitness must be calculated

        Returns
        -------
        float
            The negative of the calculated energy
        """
        if 'data' in candidate.genes.attributes:
            data = candidate.genes.attributes.pop('data')
            return - sum([i[1] for i in data.scfvalues[0]])
        if 'was_optg' in candidate.genes.attributes and candidate.genes.attributes['was_optg']:
            candidate.genes.attributes.pop('was_optg')
            return - sum([i[0] for i in data.geovalues]) + sum([i[1] for i in data.scfvalues[0]])
        data = candidate.genes.get_molpro_output(candidate.label, self.save_directory + '/data', False,
            self.threads_per_calc, False)
        return - sum([i[1] for i in data.scfvalues[0]])

    @staticmethod
    def particle_permutation(parent: Molpro_Molecule, attributes: dict) -> Molpro_Molecule:
        """Randomly changes the positions of two atoms

        Parameters
        ----------
        parent : Molpro_Molecule
            Molecule which will suffer the particle permutation
        attributes : dict
            Global parameters

        Returns
        -------
        Molpro_Molecule
            Mutated molecule
        """
        index0 = random.randint(0, len(parent.elements) - 1)
        index1 = random.choice([n for n, element in enumerate(parent.elements) if element != parent.elements[index0]])
        return particle_permutation(parent, (index0, index1))

    @staticmethod
    def piece_displacement(parent: Molpro_Molecule, attributes: dict) -> Molpro_Molecule:
        """Randomly displaces a piece of the molecule

        Parameters
        ----------
        parent : Molpro_Molecule
            Molecule which will suffer the piece displacement
        attributes : dict
            Global parameters

        Returns
        -------
        Molpro_Molecule
            Mutated molecule
        """
        direction = np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])
        return piece_displacement(parent, _piece_select(parent, attributes['piece_size_range']),
            (direction/(np.linalg.norm(direction))) * _random_displacement(attributes['displacement_range']))

    @staticmethod
    def particle_displacement(parent: Molpro_Molecule, attributes: dict) -> Molpro_Molecule:
        """Randomly displaces an atom of the molecule

        Parameters
        ----------
        parent : Molpro_Molecule
            Molecule which will suffer the particle displacement
        attributes : dict
            Global parameters

        Returns
        -------
        Molpro_Molecule
            Mutated molecule
        """
        direction = np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])
        return piece_displacement(parent, [random.randint(0, len(parent.elements) - 1)],
            (direction/(np.linalg.norm(direction))) * _random_displacement(attributes['displacement_range']))

    @staticmethod
    def piece_rotation(parent: Molpro_Molecule, attributes: dict) -> Molpro_Molecule:
        """Randomly rotates a piece of the molecule

        Parameters
        ----------
        parent : Molecule
            Molecule which will suffer the piece rotation
        attributes : dict
            Global parameters

        Returns
        -------
        Molpro_Molecule
            Mutated molecule
        """
        indexes = _piece_select(parent, attributes['piece_size_range'])
        return piece_rotation(parent, indexes, sum([parent.newcoords[i] for i in indexes])/len(indexes),
            _random_rotation(attributes['rotation_range']))

    @staticmethod
    def self_mirroring(parent: Molpro_Molecule, attributes: dict) -> Molpro_Molecule:
        """Reflects a piece of the molecule with a reflection plane which contains the geometric center of such
        piece and has a the vector which points from the piece center to the molecule center as normal direction. The
        image piece replaces the object piece

        Parameters
        ----------
        parent : Molpro_Molecule
            Molecule which piece will suffer the mirroring
        attributes : dict
            Global parameters

        Returns
        -------
        Molpro_Molecule
            Mutated molecule
        """
        indexes = _piece_select(parent, attributes['piece_size_range'])
        piece_center = sum([parent.newcoords[i] for i in indexes])/len(indexes)
        return piece_reflection(parent, indexes, (sum(parent.newcoords)/len(parent.newcoords)) - piece_center,
            piece_center, True)

    @staticmethod
    def mirroring(parent: Molpro_Molecule, attributes: dict) -> Molpro_Molecule:
        """Randomly reflects a piece of the molecule with a reflection plane which contains the geometric center of the
        molecule and has the vector which points to the geometric center of such piece as normal. The image piece
        replaces nearests compatible atoms to the geometric center of the image

        Parameters
        ----------
        parent : Molpro_Molecule
            Molecule which piece will suffer the mirroring
        attributes : dict
            Global parameters

        Returns
        -------
        Molpro_Molecule
            Mutated molecule

        Raises
        ------
        Exception
            Not enough atoms to be reflected
        Exception
            Not enough atoms to be replaced
        """
        selector_point = np.array([random.uniform(a, b) for a, b in np.array([np.array([min([parent.newcoords[i][j]
        for i in range(len(parent.newcoords))]), max([parent.newcoords[i][j] for i
            in range(len(parent.newcoords))])]) for j in range(3)])])
        indexes = [i for i in range(len(parent.newcoords))]
        indexes.sort(key=lambda i: sqrt((selector_point[0] - parent.newcoords[i][0]) ** 2 + (selector_point[1] -\
            parent.newcoords[i][1]) ** 2 + (selector_point[2] - parent.newcoords[i][2]) ** 2))
        piece_lenght = random.randint(attributes['piece_size_range'][0], attributes['piece_size_range'][1])
        piece_indexes = []
        for i in indexes:
            if piece_indexes.count(parent.elements[i]) < floor(parent.elements.count(parent.elements[i])/2):
                piece_indexes.append(i)
            if len(piece_indexes) == piece_lenght:
                break
        if len(piece_indexes) != piece_lenght:
            raise Exception('Not enough atoms to be reflected')
        for i in piece_indexes:
            indexes.remove(i)
        particle_center = sum(parent.newcoords)/len(parent.newcoords)
        object_center = sum([parent.newcoords[i] for i in piece_indexes])/len(piece_indexes)
        destiny_point = 2 * particle_center - object_center
        indexes.sort(key=lambda i: sqrt((destiny_point[0] - parent.newcoords[i][0]) ** 2 + (destiny_point[1] -\
            parent.newcoords[i][1]) ** 2 + (destiny_point[2] - parent.newcoords[i][2]) ** 2))
        destiny_indexes = []
        needed_elements = [parent.elements[i] for i in piece_indexes]
        for i in indexes:
            if parent.elements[i] in needed_elements:
                destiny_indexes.append(i)
            if len(destiny_indexes) == piece_lenght:
                break
        if len(destiny_indexes) != piece_lenght:
            raise Exception('Not enough atoms to be replaced')
        return piece_reflection(parent, piece_indexes, destiny_point - object_center, (object_center + destiny_point)/2,
            False, replacing_indexes=destiny_indexes)


    @staticmethod
    def enlarge(parent: Molpro_Molecule, attributes: dict) -> Molpro_Molecule:
        """Randomly enlarges the molecule

        Parameters
        ----------
        parent : Molpro_Molecule
            Molecule which will be enlarged
        attributes : dict
            Global parameters

        Returns
        -------
        Molpro_Molecule
            Mutated molecule
        """
        return enlarge_reduce(parent, random.uniform(1, attributes['enlarge_reduce_range'][1]))

    @staticmethod
    def reduce(parent: Molpro_Molecule, attributes: dict) -> Molpro_Molecule:
        """Randomly reduces the molecule

        Parameters
        ----------
        parent : Molpro_Molecule
            Molecule which will be reduced
        attributes : dict
            Global parameters

        Returns
        -------
        Molpro_Molecule
            Mutated molecule
        """
        return enlarge_reduce(parent, random.uniform(attributes['enlarge_reduce_range'][0], 1))

    @staticmethod
    def piece_crossover(parent: Molpro_Molecule, donor: Molpro_Molecule, attributes: dict) -> Molpro_Molecule:
        """Produces a child molecule using the parent's geometry in which one piece will be replaced by one of the donor

        Parameters
        ----------
        parent : Molpro_Molecule
            Parent molecule
        donor : Molecule
            Donor molecule
        attributes : dict
            Global parameters

        Returns
        -------
        Molpro_Molecule
            Child molecule

        Raises
        ------
        Exception
            Not enough atoms to be replaced
        """
        child = parent.copy()
        piece_indexes = _piece_select(donor, attributes['piece_size_range'])
        piece = np.array([donor.newcoords[i] for i in piece_indexes])
        insertion_point = sum(piece)/len(piece)
        indexes = [i for i in range(len(parent.newcoords))]
        indexes.sort(key=lambda i: sqrt((insertion_point[0] - parent.newcoords[i][0]) ** 2 + (insertion_point[1] -\
            parent.newcoords[i][1]) ** 2 + (insertion_point[2] - parent.newcoords[i][2]) ** 2))
        destiny_indexes = []
        needed_elements = [donor.elements[i] for i in piece_indexes]
        for i in indexes:
            if parent.elements[i] in needed_elements:
                destiny_indexes.append(i)
            if len(destiny_indexes) == len(piece_indexes):
                break
        if len(destiny_indexes) != len(piece_indexes):
            raise Exception('Piece crossover lenght error')
        for n, i in enumerate(destiny_indexes):
            child.newcoords[i] = piece[n]
        child.build_zmatrix()
        return child

    @staticmethod
    def randomize(attributes: dict) -> Molpro_Molecule:
        """Returns a molecule with an entirely random geometry

        Parameters
        ----------
        attributes : dict
            Global parameters

        Returns
        -------
        Molpro_Molecule
            Molecule with random geometry
        """
        return randomize(attributes['first_candidate'].genes, attributes['distance_range'])

    def local_optimize(self, candidate: Chromosome) -> Molpro_Molecule:
        """Executes geometric optimization over the candidate's molecule using optg and returns a new molecule with the
        optimized geometry

        Parameters
        ----------
        candidate : Chromosome
            Candidate which genes will be locally optimized

        Returns
        -------
        Molpro_Molecule
            Locally optimized molecule
        """
        return candidate.genes.optg(candidate.label, self.save_directory + '/data', 
            nthreads = self.pool_size * self.threads_per_calc)

    def catch(self, candidate: Chromosome) -> None:
        """Method which will be executed if an error occurs during a candidate production

        Parameters
        ----------
        candidate : Chromosome
            Candidate in which production some exception occurred
        """
        os.remove(f'{self.save_directory}/data/{candidate.label}.inp')
        os.remove(f'{self.save_directory}/data/{candidate.label}.out')
        os.remove(f'{self.save_directory}/data/{candidate.label}.xml')

    @staticmethod
    def save(candidate: Chromosome, filename: str, directory: str) -> None:
        """Saves the candidate data in a .inp document

        Parameters
        ----------
        candidate : Chromosome
            Candidate which data will be saved
        filename : str
            Document's name
        directory : str
            Directory where the document will be saved
        """
        candidate.genes.save(filename, directory)


def _piece_select(molecule: Molpro_Molecule, piece_size_range: Tuple[int, int]) -> list[int]:
    """Randomly selects a piece of the molecule

    Parameters
    ----------
    molecule : Molpro_Molecule
        Molecule from which a piece will be selected
    piece_size_range : Tuple[int, int]
        Range in which pieces will be picked

    Returns
    -------
    list[int]
        Piece indexes
    """
    selector_point = np.array([random.uniform(a, b) for a, b in np.array([np.array([min([molecule.newcoords[i][j]
        for i in range(len(molecule.newcoords))]), max([molecule.newcoords[i][j] for i
            in range(len(molecule.newcoords))])]) for j in range(3)])])
    indexes = [i for i in range(len(molecule.newcoords))]
    indexes.sort(key=lambda i: sqrt((selector_point[0] - molecule.newcoords[i][0]) ** 2 + (selector_point[1] -\
        molecule.newcoords[i][1]) ** 2 + (selector_point[2] - molecule.newcoords[i][2]) ** 2))
    return indexes[:random.randint(piece_size_range[0], piece_size_range[1])]


def _random_displacement(displacement_range: Tuple[float, float]) -> np.ndarray[float]:
    """Produces a random displacement vector

    Parameters
    ----------
    displacement_range : Tuple[float, float]
        Range in which particles can be randomly displaced

    Returns
    -------
    np.ndarray[float]
        Displacement vector
    """
    return random.uniform(displacement_range[0], displacement_range[1])


def _random_rotation(rotation_range: Tuple[float, float]) -> np.ndarray[float, float, float]:
    """Produces a random 3d rotation array

    Parameters
    ----------
    rotation_range : Tuple[float, float]
        Range in which particles can be randomly rotated

    Returns
    -------
    np.ndarray[float]
        3d rotation array
    """
    return np.array([
        random.uniform(rotation_range[0], rotation_range[1]), random.uniform(rotation_range[0], rotation_range[1]),
        random.uniform(rotation_range[0], rotation_range[1])
    ])
