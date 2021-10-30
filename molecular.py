import re
import random
import os
import copy
from typing import Union, Any


"""Framework for dealing with molecular structures and genetic algorithm compatible with Molpro
"""


numeric = Union[float, int]


class Molecule:
    """Molecular geometry compatible with Molpro

    This class provides a framework for storing molecular geometry, generating Molpro inputs and store output 
    information. It also brings functions to work with genetic algorithm.
    """
    Molecule = Any
    __slots__ = ('basis', 'parameters', 'geometry', 'settings', 'rand_range', 'label', 'output_values', 'output',
        'optg_result', 'was_optg')
    
    def __init__(self, basis: str, geometry: list[list[str]], settings: list[str], parameters: dict=dict(), 
        rand_range: tuple[numeric, numeric] = None, label: str = None, output: str = None, output_values: dict = dict(),
        was_optg: bool = False):
        """Initializes the Molecule object by receiving its parameters
        
        :param basis: Hilbert space basis
        :type basis: str
        :param geometry: Z-matrix input
        :type geometry: list[list[str]]
        :param settings: Molpro calculation settings
        :type settings: list[str]
        :param parameters: The values of geometry variables, defaults to dict()
        :type parameters: dict, optional
        :param rand_range: Min and max values that can be generated when distances are mutated, defaults to None
        :type rand_range: tuple[numeric, numeric], optional
        :param label: A tag which can be used to identify the object, defaults to None
        :type label: str, optional
        :param output: The address of the molecule's .out document generated by Molpro, defaults to None
        :type output: str, optional
        :param output_values: Values extracted from the output document, defaults to dict()
        :type output_values: dict, optional
        :param was_optg: True if the geometry was already optmized with optg, false if it doesn't
        :type was_optg: bool, optional
        """        
        self.basis = basis
        self.parameters = parameters
        self.geometry = geometry
        self.settings = settings
        self.rand_range = rand_range
        self.label = label
        self.output = output
        self.output_values = output_values
        self.was_optg = was_optg

    @property
    def dist_unit(self) -> str:
        """Gets the distance unit used to describe the molecule

        :return: Distance unit
        :rtype: str
        """        
        return self.geometry[0][0]

    def __str__(self) -> str:
        """Returns the Molpro input string

        :return: Molpro input string
        :rtype: str
        """        
        string = f'***,\n\nbasis={self.basis}\n\n'
        if len(self.parameters) > 0:
            for k, v in zip(self.parameters.keys(), self.parameters.values()):
                string += f'{k}={v}\n'
        else:
            string = string[:-1]
        string += '\ngeometry={'
        for row in self.geometry:
            string += str(row).replace('[', '').replace(']', '').replace('\'', '') + '\n'
        string += '}\n\n'
        for setting in self.settings:
            string += setting + '\n'
        string += '\n---'
        return string
    
    def save(self, document: str = None, directory: str = 'data') -> None:
        """Saves the object data in a .inp document

        Saves the object data in a [document].inp. If [document] receives None it'll be the molecule's label, if it 
        still None it will be str(abs([molecule].__hash__())). If the directory /[directory] didn't exist it wil be
        created

        :param document: Document's name, defaults to None
        :type document: str, optional
        :param directory: Directory where the document will be saved, defaults to 'data'
        :type directory: str, optional
        """              
        document = document if document is not None else self.label if\
             self.label is not None else str(abs(self.__hash__()))
        if directory == '':
            with open(f'{document}.inp', 'w') as file:
                file.write(str(self))
        else:
            if not os.path.exists(directory):
                os.mkdir(directory)
            with open(f'{directory}/{document}.inp', 'w') as file:
                file.write(str(self))

    def get_value(self, wanted: list[str], document: str = None, directory: str = 'data', keep_output: bool = False, 
        nthreads: int = 1, update_self: bool = True) -> dict:
        """Reads the Molpro's output file and return the wanted values.

        Reads the Molpro's output file, searchs for wanted strings and gets the numeric value that is in the same row
        then returns a dictionary where the keys are the wanted strings and the values are the numeric strings 
        correspondents. If [document].out already exists in [directory] the document will be read and the values
        returned. If it doesn't and [document].inp already exists in directory, it will execute Molpro over the input.
        If neither [document].out nor [document].inp exists, [document].inp will be created and Molpro executed over it
        and after [document].inp will be deleted

        :param wanted: list of variables to be search in the output
        :type wanted: list[str]
        :param document: Documents' name, defaults to None
        :type document: str, optional
        :param directory: Directory adress where the input and the output will be relative to __main__, defaults to 
            'data'
        :type directory: str, optional
        :param keep_output: If it's True, the output will be kept and its name will be put in [self.output], else it 
            will be deleted after is read. defaults to False
        :type keep_output: bool, optional
        :param nthreads: Number of threads useds in Molpro calculation, defaults to 1
        :type nthreads: int, optional
        :param update_self: If it's True, [self.output_values] will be updated with each item of wanted. If it's False, 
            [self.output_values] will not be updated, and the output values will can only be accessed by the returned 
            dict
        :type update_self: bool, optional
        :return: Wanted strings and correspondent values
        :rtype: dict
        """        
        document = document if document is not None else self.label if self.label is not None else str(self.__hash__())
        deldoc = False
        output_dictionary = self.output_values if update_self else dict()
        if not os.path.exists(f'{directory}/{document}.out'):
            if not os.path.exists(f'{directory}/{document}.inp'):
                self.save(document, directory)
                deldoc = True
            os.system(f"molpro -n {nthreads} './{directory}/{document}.inp'")
        with open(f'{directory}/{document}.out', 'r') as file:
            outstr = file.read()
        for item in wanted:
            output_dictionary.update({item: re.search('-*[0-9.]+', re.search(f'{item}.*', outstr)[0]\
                .replace(item, ''))[0]})
        if deldoc:
            os.remove(f'{directory}/{document}.inp')
        if not keep_output:
            os.remove(f'{directory}/{document}.xml')
            os.remove(f'{directory}/{document}.out')
        else:
            self.output = f'{directory}/{document}.out'
        return output_dictionary

    def optg(self, wanted: list[str], directory: str = 'data', nthreads: int = 1, keep_output= False) -> Molecule:
        """Turns the molecule in its own geometric optimized version

        :param wanted: list of variables to be search in the output
        :type wanted: list[str]
        :param directory: Directory adress where the input and the output will be relative to __main__, defaults to 
            'data'
        :type directory: str, optional
        :param nthreads: Number of threads useds in Molpro calculation, defaults to 1
        :type nthreads: int, optional
        :param keep_output: If it's True, the output will be kept, else it will be deleted after be read. 
            defaults to False
        :type keep_output: bool, optional
        :return: Itself optmized version
        :rtype: Molecule
        """        
        self._receive(optg(self, wanted, directory, nthreads, keep_output))
        return self
    
    def copy(self) -> Molecule:
        """Returns a totally independent copy of itself

        :return: Copy of self
        :rtype: Molecule
        """        
        return copy.deepcopy(self)
    
    def swap_mutate(self) -> Molecule:
        """Provokes a swap mutation in itsef

        :return: Itself mutated
        :rtype: Molecule
        """        
        self._receive(swap_mutate(self))
        return self

    def _receive(self, molecule) -> None:
        """
        Receives all data from another molecule making the actual molecule its copy

        :param molecule: The data donor molecule
        :type molecule: Molecule
        """        
        self.basis = molecule.basis
        self.parameters = molecule.parameters
        self.geometry = molecule.geometry
        self.settings = molecule.settings
        self.rand_range = molecule.rand_range
        self.label = molecule.label
        self.output = molecule.output
        self.output_values = molecule.output_values

    def mutate_distances(self, times: int = 1) -> Molecule:
        """Provokes a mutation in some random distance parameter

        Selects a random distance parameter from the geometry and assign it a random value in the range gave by 
        [self.rand_range]. This process is repeated a number of times equal to [times]

        :param times: Number of mutations, defaults to 1
        :type times: int
        :return: Itself mutated
        :rtype: Molecule
        """        
        self._receive(mutate_distances(self, times))
        return self
    
    def mutate_angles(self, times: int = 1) -> Molecule:
        """Provokes a mutation in some random angle parameter
        
        Selects a random angle parameter from the geometry and assign it a random value between 0 and 
        360 degrees. This process is repeated an amount of times equal to [times]

        :param times: Number of mutations, defaults to 1
        :type times: int, optional
        :return: Itself mutated
        :rtype: Molecule
        """        
        self._receive(mutate_angles(self, times))
        return self

    @staticmethod
    def load(file: str, rand_range: tuple[numeric, numeric] = None, label: str = None, output: str = None, 
        output_values: dict = dict(), was_optg: bool = False) -> Molecule:
        """Loads a molecule from a .inp document and returns its Molecule object

        :param file: Fille name
        :type file: str
        :param rand_range: Min and max values that can be generated when distances are mutated, defaults to None
        :type rand_range: tuple[numeric, numeric], optional
        :param label: A tag which can be used to identify the object, defaults to None
        :type label: str, optional
        :param output: The address of the molecule's .out document generated by Molpro, defaults to None
        :type output: str, optional
        :param output_values: Values extracted from the output document, defaults to dict()
        :type output_values: dict, optional
        :raises Exception: Invalid Z-matrix
        :return: Loaded molecule
        :rtype: Molecule
        """        
        with open(file, 'r') as data:
            inpstr = re.search('\*\*\*,.*---', data.read(), flags=re.S)[0]
        basis = re.search('basis=.*', inpstr, flags=re.S)[0].split('\n\n')[0].replace('basis=', '')
        splitinpstr = inpstr.split('\n\n')
        leninspstr = len(splitinpstr)
        if leninspstr == 6:
            parameters = {p.split('=')[0]:p.split('=')[1] for p in splitinpstr[2].split('\n')}
            settings_location = 4
        elif leninspstr == 5:
            parameters = dict()
            settings_location = 3
        else:
            raise Exception('Invalid input format.')
        geometry = [re.split(' *, *', row) for row in re.search('geometry=.*', inpstr, flags=re.S)[0].split('\n\n')[0].\
                    replace('geometry=', '').replace('{', '').replace('}', '').split('\n')]
        geometry.remove([''])
        settings = splitinpstr[settings_location].split('\n')
        return Molecule(basis, geometry, settings, parameters, rand_range, label, output, output_values)
    

def swap_mutate(molecule: Molecule, times: int = 1, label: str = None) -> Molecule:
    """Returns a molecule's copy with random swapped places parameters

    Creates a molecule's copy and randomly swap parameters of its places. It makes a amount of random swaps equal 
    [times].

    :param molecule: Original molecule
    :type molecule: Molecule
    :param times: Number of random swaps, defaults to 1
    :type times: int, optional
    :param label: Label of the new molecule, defaults to None
    :type label: str, optional
    :return: New molecule
    :rtype: Molecule
    """    
    new_molecule = molecule.copy()
    for _ in range(times):
        index0, index1, index2, index3 = 0, 0, 0, 0
        while (index0, index1) == (index2, index3):
            index0, index1, index2, index3 = _get_swap_indexes(new_molecule)
        new_molecule.geometry[index0][index1], new_molecule.geometry[index2][index3] =\
        new_molecule.geometry[index2][index3], new_molecule.geometry[index0][index1]
    new_molecule.output = None
    new_molecule.output_values = dict()
    new_molecule.label = label
    new_molecule.was_optg = False
    return new_molecule


def _get_swap_indexes(new_molecule:Molecule) -> tuple[int, int, int, int]:
    """Randomly choices the indexes of the parameters that will be swapped

    :param new_molecule: Copy of the original moecule
    :type new_molecule: Molecule
    :return: A tuple with the four indexes needed to perform the swap
    :rtype: tuple[int, int, int, int]
    """    
    index0 = random.choice(range(2, len(new_molecule.geometry)))
    index1 = random.choice(range(2, len(new_molecule.geometry[index0]) + 1, 2))
    if index1 == 2:
        index2 = random.choice(range(2, len(new_molecule.geometry)))
        index3 = 2
    else:
        index2 = random.choice(range(3, len(new_molecule.geometry)))
        index3 = random.choice(range(4, len(new_molecule.geometry[index2]) + 1, 2))
    return (index0, index1, index2, index3)


def mutate_angles(molecule: Molecule, times: int = 1, label: str = None) -> Molecule:
    """Returns a molecule's copy with some random angle parameter randomized between 0 and 360 degrees

    Creates a moecule's copy and randomly choices an angle parameter and set it value to a random value between 0 and
    360. The amount of mutations performed is equal [times]

    :param molecule: Original molecule
    :type molecule: Molecule
    :param times: Amount of angle mutations, defaults to 1
    :type times: int, optional
    :param label: Label of the new molecule, defaults to None
    :type label: str, optional
    :return: New molecule
    :rtype: Molecule
    """    
    new_molecule = molecule.copy()
    for _ in range(times):
        index0 = random.choice(range(3, len(new_molecule.geometry)))
        index1 = random.choice(range(4, len(new_molecule.geometry[index0]) + 1, 2))
        if new_molecule.geometry[index0][index1].replace('.', '').isdigit():
            new_molecule.geometry[index0][index1] = str(random.uniform(0, 360))
        else:
            new_molecule.parameters[new_molecule.geometry[index0][index1]] = str(random.uniform(0, 360))
    new_molecule.output = None
    new_molecule.output_values = dict()
    new_molecule.label = label
    new_molecule.was_optg = False
    return new_molecule


def mutate_distances(molecule: Molecule, times: int = 1, label: str = None) -> Molecule:
    """Returns a molecule's copy with some random distance parameter randomized in the range gave by [self.rand_range]

    Creates a moecule's copy and randomly choices a distance parameter and set it value to a random value between 0 and
    [self.rand_range]. The amount of mutations performed is equal [times]

    :param molecule: Original molecule
    :type molecule: Molecule
    :param times: Amount of angle mutations, defaults to 1
    :type times: int, optional
    :param label: Label of the new molecule, defaults to None
    :type label: str, optional
    :return: New molecule
    :rtype: Molecule
    """    
    new_molecule = molecule.copy()
    for _ in range(times):
        index0 = random.choice(range(2, len(new_molecule.geometry)))
        if new_molecule.geometry[index0][2].replace('.', '').isdigit():
            new_molecule.geometry[index0][2] = str(random.uniform(new_molecule.rand_range[0], 
                new_molecule.rand_range[1]))
        else:
            new_molecule.parameters[new_molecule.geometry[index0][2]] = str(random.uniform(new_molecule.rand_range[0], 
                new_molecule.rand_range[1]))
    new_molecule.output = None
    new_molecule.output_values = dict()
    new_molecule.label = label
    new_molecule.was_optg = False
    return new_molecule


def random_molecule(molecular_formula: str, basis: str, settings: list[str], rand_range: tuple[numeric, numeric], 
    label: str = None, dist_unit: str = 'ang') -> Molecule:
    """Creates molecule with a entirely random geometry given the molecular formula

    Creates a geometry with all distances parameters randomized in the range gave by [rand_range] and all angles 
    randomized between 0 and 360 given the molecular formula which is a string containing the elements followed by its 
    amount like'H2O', 'C6H12O6', 'Al10'... It doesn't support parenthesys and is invariant to the order of elements. 
    What matters in [molecular_formula] is which number follows which element

    :param molecular_formula: The molecular formula of the wanted molecule
    :type molecular_formula: str
    :param basis: Hilbert space basis
    :type basis: str
    :param settings: Molpro calculation settings
    :type settings: list[str]
    :param rand_range: Min and max values that can be generated when distances are mutated
    :type rand_range: tuple[numeric, numeric]
    :param label: A tag which can be used to identify the molecule, defaults to None
    :type label: str, optional
    :param dist_unit: Distance unit, defaults to 'ang'
    :type dist_unit: str, optional
    :return: Random molecule
    :rtype: Molecule
    """    
    atoms = re.findall('[A-Z][a-z]*[1-9]*', molecular_formula)
    geometry = [[dist_unit]]
    for atom in atoms:
        element = re.sub('[0-9]*', '', atom)
        times = re.sub('[A-Z][a-z]*', '', atom)
        if times:
            times = int(times)
        else:
            times = 1
        for t in range(times):
            geometry.append([element]) 
            if len(geometry) > 2:
                geometry[-1].append('1')
                geometry[-1].append(str(random.uniform(rand_range[0], rand_range[1])))
                if len(geometry) > 3:
                    for n in (2, 3):
                        geometry[-1].append(str(n))
                        geometry[-1].append(str(random.uniform(0, 360)))
    return Molecule(basis, geometry, settings, rand_range=rand_range, label=label)


def crossover_n(parent: Molecule, donor: Molecule, label: str = None) -> Molecule:
    """Returns a new molecule which randomly carries parameters from the parent and donor molecules.

    Creates a copy of the parent molecule, randomly choices an amount of parameters the minimun being one and the 
    maximun being the total amount of parameters minus one. Then randomly choices this amount of parameters from the 
    donor molecule to replace the respectives parameters the child molecule inherited from parent molecule.

    :param parent: Parent molecule
    :type parent: Molecule
    :param donor: Donor molecule
    :type donor: Molecule
    :param label: A tag which can be used to identify the child molecule, defaults to None
    :type label: str, optional
    :return: Child molecule
    :rtype: Molecule
    """    
    child = parent.copy()
    child.label = label
    if len(child.parameters) > 0:
        n_parameters = random.randint(1, len(child.parameters) - 1)
    else:
        n_parameters = 0
    n_atoms = random.randint(1, len(child.geometry) - 3)
    parameters_indexes = []
    atoms_indexes = []
    for _ in range(n_parameters):
        index = random.randint(0, len(child.parameters) - 1)
        while index in parameters_indexes:
            index = random.randint(0, len(child.parameters) - 1)
        parameters_indexes.append(index)
        key = list(child.parameters.keys())[index]
        child.parameters[key] = donor.parameters[key]
    for _ in range(n_atoms):
        index = random.randint(2, len(child.geometry) - 1)
        while index in atoms_indexes:
            index = random.randint(2, len(child.geometry) - 1)
        atoms_indexes.append(index)
        child.geometry[index] = donor.geometry[index]
    child.output = None
    child.output_values = dict()
    child.was_optg = False
    return child


def randomize(molecule: Molecule, label: str = None) -> Molecule:
    """Returns a molecule's copy with all distances and angles parameters randomized

    Creates a molecule's copy and replace all distance parameters with random values in the range gave by 
    [molecule.rand_range] and replace all angle parameters with random values between 0 and 360

    :param molecule: Original molecule
    :type molecule: Molecule
    :param label: A tag which can be used to identify the new molecule, defaults to None
    :type label: str, optional
    :return: New molecule
    :rtype: Molecule
    """    
    new_molecule = molecule.copy()
    new_molecule.label = label
    for row_index in range(2, len(new_molecule.geometry)):
        if new_molecule.geometry[row_index][2].replace('.', '').isdigit():
            new_molecule.geometry[row_index][2] = str(random.uniform(new_molecule.rand_range[0], 
                new_molecule.rand_range[1]))
        else:
            new_molecule.parameters[new_molecule.geometry[row_index][2]] =\
                str(random.uniform(new_molecule.rand_range[0], new_molecule.rand_range[1]))
        for angle_index in range(4, len(new_molecule.geometry[row_index]), 2):
            if new_molecule.geometry[row_index][angle_index].replace('.', '').isdigit():
                new_molecule.geometry[row_index][angle_index] = str(random.uniform(0, 360))
            else:
                new_molecule.parameters[new_molecule.geometry[row_index][angle_index]] = str(random.uniform(0, 360))
    new_molecule.output = None
    new_molecule.output_values = dict()
    new_molecule.was_optg = False
    return new_molecule


def crossover_1(parent: Molecule, donor: Molecule, label: str = None) -> Molecule:
    """Produces a new molecule with the crossover of parent and donor molecules by cutting each one in one point and
    combining the resultant pieces.

    Creates a copy of the parent molecule then randomly choices a row and a 'column' from the geometry and there divides
    the geometry in two pieces. Then randomly pick one of these pieces and attach with the complementar part provided by
    the donor molecule generating the geometry of the child molecule.

    :param parent: Parent molecule
    :type parent: Molecule
    :param donor: Donor molecule
    :type donor: Molecule
    :param label: A tag which can be used to identify the child molecule, defaults to None
    :type label: str, optional
    :return: Child molecule
    :rtype: Molecule
    """    
    child = parent.copy()
    keys = list(child.parameters.keys())
    child.label = label
    index = random.choice(range(2, len(child.geometry)))
    if random.choice([True, False]):
        child.geometry[index:] = donor.geometry[index:]
    else:
        child.geometry[:index] = donor.geometry[:index]
    if len(child.parameters) > 0:
        index = random.choice(range(0, len(child.parameters)))
        if random.choice([True, False]):
            for parameter in keys[index:]:
                child.parameters[parameter] = donor.parameters[parameter]
        else:
            for parameter in keys[:index]:
                child.parameters[parameter] = donor.parameters[parameter]
    child.output = None
    child.output_values = dict()
    child.was_optg = False
    return child


def crossover_2(parent: Molecule, donor: Molecule, label: str = None) -> Molecule:
    """Produces a new molecule with the crossover of parent and donor molecules by cutting each one in two points and
    combining the resultant pieces.
    
    Randomly cuts the parent molecule's geometry in two points. The sequence between these two points will either 
    replace its correspondent in the donor molecule's geometry or be replaced by it (randomly) generating the geometry
    of the child molecule that will be returned. At leas four atoms are needed to realize this process. Trying it with
    molecules smaller than it will raise an exception.

    :param parent: Parent molecule
    :type parent: Molecule
    :param donor: Donor molecule
    :type donor: Molecule
    :param label: A tag which can be used to identify the child molecule, defaults to None
    :type label: str, optional
    :raises Exception: At least 4 atoms are needed to perform crossover_2
    :return: child Molecule
    :rtype: Molecule
    """    
    if len(parent.geometry) < 5:
        raise Exception('Molecules does not has enough atoms for crossover_2.')
    child = parent.copy()
    child.label = label
    child.output = None
    child.output_values = dict()
    index1 = random.choice(range(2, len(child.geometry)))
    index2 = random.choice(range(2, len(child.geometry)))
    while index1 == index2 or ((index1 == 2 and index2 == len(child.geometry) - 1) \
        or (index1 == len(child.geometry) - 1 and index2 == 2)):
        index1 = random.choice(range(2, len(child.geometry)))
        index2 = random.choice(range(2, len(child.geometry)))
    if index1 > index2:
        index1, index2 = index2, index1
    if random.choice([True, False]):
        child.geometry[:index1] = donor.geometry[:index1]
        child.geometry[index2:] = donor.geometry[index2:]
    else:
        child.geometry[index1:index2] = donor.geometry[index1:index2]
    if len(child.parameters) > 0:
        keys = list(child.parameters.keys())
        index1 = random.choice(range(0, len(child.parameters)))
        index2 = random.choice(range(0, len(child.parameters)))
        if index1 > index2:
            index1, index2 = index2, index1
        if random.choice([True, False]):
            for parameter in keys[:index1] + keys[index2:]:
                child.parameters[parameter] = donor.parameters[parameter]
        else:
            for parameter in keys[index1:index2]:
                child.parameters[parameter] = donor.parameters[parameter]
    child.was_optg = False
    return child


def optg(molecule: Molecule, wanted_energy: str, directory: str = 'data', nthreads: int = 1, 
    keep_output: bool = False) -> Molecule:
    """Executes geometric optimization over the molecule using Molpro and returns a new molecule with the optimized 
    geometry

    Creates a copy of the original molecule, appends 'optg' in the settings if it's not already there and calls the 
    get_value (see with 'help(get_value)') function with it. Then updates the [wanted_energy] in 
    [optmized_molecule.output_values] and updates all angles and distances parameters in [optmized_molecule.parameters].
    To use this function there cannont be any literal value in the molecule's geometry, all values must be as variables
    names in geometry and has its literal values declared in [molecule.parameters].

    :param molecule: Original molecule
    :type molecule: Molecule
    :param wanted_energy: The key wanted for the energy in [optimized_molecule.output_values]
    :type wanted_energy: str
    :param directory: Directory adress where the input and the output will be relative to __main__, defaults to 'data'
    :type directory: str, optional
    :param nthreads: Number of threads useds in Molpro calculation, defaults to 1
    :type nthreads: int, optional
    :param keep_output: If it's True, the output will be kept and its name will be put in [self.output], else it will be 
        deleted after is read. defaults to False
    :type keep_output: bool, optional
    :return: Optmized molecule
    :rtype: Molecule
    """    
    opt_molecule = molecule.copy()
    if molecule.label is None:
        opt_molecule.label = str(opt_molecule.__hash__())
    opt_molecule.label += '_optg'
    if not 'optg' in opt_molecule.settings:
        opt_molecule.settings.append('optg')
    if not 'total_energy = energy' in opt_molecule.settings:
        opt_molecule.settings.append('total_energy = energy')
    opt_molecule.output_values.update({wanted_energy: opt_molecule.get_value(['TOTAL_ENERGY'], keep_output=True, 
        nthreads=nthreads, update_self=False)['TOTAL_ENERGY']})
    with open(f'{directory}/{opt_molecule.label}.out', 'r') as file:
        outstr = file.read()
        for parameter in opt_molecule.parameters.keys():
            opt_molecule.parameters[parameter] = re.search('-*[0-9.]+', re.sub(parameter, '', 
                re.findall(f'{parameter}=.*', outstr, flags=re.I)[1], flags=re.I))[0]
    opt_molecule.settings.remove('optg')
    opt_molecule.settings.remove('total_energy = energy')
    if not keep_output:
        os.remove(f'{directory}/{opt_molecule.label}.log')
        os.remove(f'{directory}/{opt_molecule.label}.xml')
        os.remove(f'{directory}/{opt_molecule.label}.out')
    else:
        opt_molecule.output = f'{directory}/{opt_molecule.label}.out'
    if molecule.label is None:
        opt_molecule.label = None
    opt_molecule.was_optg = True
    return opt_molecule
