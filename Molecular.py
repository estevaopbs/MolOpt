import re
import random
import os
import copy


class Molecule:
    __slots__ = ('basis', 'parameters', 'geometry', 'settings', 'rand_range', 'label', 'output_values', 'output')
    def __init__(self, basis:str, geometry:list, settings:list, parameters:dict=dict(), rand_range:float=None, 
                 label:str=None, output:str=None, output_values:dict=dict()):
        self.basis = basis
        self.parameters = parameters
        self.geometry = geometry
        self.settings = settings
        self.rand_range = rand_range
        self.label = label
        self.output = output
        self.output_values = output_values

    @property
    def dist_unit(self):
        return self.geometry[0][0]

    def __str__(self) -> str:
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
    
    def save(self, document:str=None, directory:str='data') -> None:
        document = document if document is not None else self.label if self.label is not None else str(abs(self.__hash__()))
        directory += '/'
        if not os.path.exists(directory):
            os.mkdir(directory)
        with open(f'{directory}{document}.inp', 'w') as file:
            file.write(str(self))

    def get_value(self, wanted:list, document=None, directory:str='data', wait:bool=True, keep_output:bool=True):
        document = document if document is not None else self.label if self.label is not None else str(self.__hash__())
        deldoc = False
        input_address = '/{directory}/{document}.inp'
        if not os.path.isfile(input_address):
            self.save(document, directory)
            deldoc = True
        output_address = f'/{directory}/{document}.out'
        if not os.path.isfile(output_address):
            os.system(f'molpro .{input_address}')
        if wait or os.path.isfile(output_address):
            while not os.path.isfile(output_address):
                continue
            with open(output_address, 'r') as output:
                outstr = output.read()
            for item in wanted:
                self.output_values.update({item: re.search(f'{item}.*', outstr)[0]})
        self.output = output_address
        if deldoc:
            os.remove(input_address)
        if not keep_output:
            os.remove(output_address)
    
    def copy(self):
        return copy.deepcopy(self)
    
    def swap_mutate(self):
        self._receive(swap_mutate(self))

    def _receive(self, molecule):
        self.basis = molecule.basis
        self.parameters = molecule.parameters
        self.geometry = molecule.geometry
        self.settings = molecule.settings
        self.rand_range = molecule.rand_range
        self.label = molecule.label
        self.output = molecule.output
        self.output_values = molecule.output_values

    def mutate_distances(self):
        self._receive(mutate_distances(self))
    
    def mutate_angles(self):
        self._receive(mutate_angles(self))

    @staticmethod
    def load(file:str, rand_range=None, label:str=None, output=None, output_values=None):
        inpstr = re.search('\*\*\*,.*---', open(file, 'r').read(), flags=re.S)[0]
        basis = re.search('basis=.*', inpstr)[0].split('=')[1]
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
        geometry = [re.split(' *, *', row) for row in re.search('{.*}', inpstr, flags=re.S)[0].split('\n')[0:-1]]
        geometry[0][0]=geometry[0][0][1:].replace('{', '').replace(' ', '')
        settings = splitinpstr[settings_location].split('\n')
        return Molecule(basis, geometry, settings, parameters, rand_range, label, output, output_values)
    

def swap_mutate(molecule) -> Molecule:
    new_molecule = molecule.copy()
    index0, index1, index2, index3 = 0, 0, 0, 0
    while (index0, index1) == (index2, index3):
        index0, index1, index2, index3 = _get_swap_indexes(new_molecule)
    new_molecule.geometry[index0][index1], new_molecule.geometry[index2][index3] =\
    new_molecule.geometry[index2][index3], new_molecule.geometry[index0][index1]
    new_molecule.output = None
    new_molecule.output_values = None
    return new_molecule


def _get_swap_indexes(new_molecule):
    index0 = random.choice(range(2, len(new_molecule.geometry)))
    index1 = random.choice(range(2, len(new_molecule.geometry[index0]), 2))
    if index1 == 2:
        index2 = random.choice(range(2, len(new_molecule.geometry)))
        index3 = 2
    else:
        index2 = random.choice(range(3, len(new_molecule.geometry)))
        index3 = random.choice(range(4, len(new_molecule.geometry[index0]), 2))
    return index0, index1, index2, index3


def mutate_angles(molecule) -> Molecule:
    new_molecule = molecule.copy()
    index0 = random.choice(range(3, len(new_molecule.geometry)))
    index1 = random.choice(range(4, len(new_molecule.geometry[index0]), 2))
    if new_molecule.geometry[index0][index1].replace('.', '').isdigit():
        new_molecule.geometry[index0][index1] = str(random.uniform(0, 180))
    else:
        new_molecule.parameters[new_molecule.geometry[index0][index1]] = str(random.uniform(0, 180))
    return new_molecule


def mutate_distances(molecule) -> Molecule:
    new_molecule = molecule.copy()
    index0 = random.choice(range(2, len(new_molecule.geometry)))
    if new_molecule.geometry[index0][2].replace('.', '').isdigit():
        new_molecule.geometry[index0][2] = str(random.uniform(0, new_molecule.rand_range))
    else:
        new_molecule.parameters[new_molecule.geometry[index0][2]] = str(random.uniform(0, new_molecule.rand_range))
    new_molecule.output = None
    new_molecule.output_values = None
    return new_molecule


def random_molecule(molecule:str, basis:str, settings:list, rand_range:float,
                    label:str=None, dist_unit:str='ang') -> Molecule:
    atoms = re.findall('[A-Z][a-z]*[1-9]*', molecule)
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
                geometry[-1].append(str(random.uniform(0, rand_range)))
                for n in range(2, (len(geometry) - 1)):
                    geometry[-1].append(str(n))
                    geometry[-1].append(str(random.uniform(0, 180)))
    return Molecule(basis, geometry, settings, rand_range=rand_range, label=label)


def crossover_n(parent:Molecule, donor:Molecule, label:str=None) -> Molecule:
    child = parent.copy()
    child.label = label
    n_parameters = random.randint(1, len(child.parameters) - 1) if len(child.parameters) > 0 else 0
    n_atoms = random.randint(1, len(child.geometry) - 3)
    parameters_indexes = []
    atoms_indexes = []
    for _ in range(n_parameters):
        index = random.randint(0, len(child.parameters))
        while index in parameters_indexes:
            index = random.randint(0, len(child.parameters))
        parameters_indexes.append(index)
        key = list(child.parameters.keys())[index]
        child.parameters[key] = donor.parameters[key]
    for _ in range(n_atoms):
        index = random.randint(2, len(child.geometry))
        while index in atoms_indexes:
            index = random.randint(2, len(child.geometry))
        atoms_indexes.append(index)
        child.geometry[index] = donor.geometry[index]
    child.output = None
    child.output_values = None
    return child


def randomize(molecule:Molecule, label:str=None) -> Molecule:
    new_molecule = molecule.copy()
    new_molecule.label = label
    for row_index in range(2, len(new_molecule.geometry)):
        if new_molecule.geometry[row_index][2].replace('.', '').isdigit():
            new_molecule.geometry[row_index][2] = str(random.uniform(0, new_molecule.rand_range))
        else:
            new_molecule.parameters[new_molecule.geometry[row_index][2]] =\
                str(random.uniform(0, new_molecule.rand_range))
        for angle_index in range(4, len(new_molecule.geometry[row_index]), 2):
            if new_molecule.geometry[row_index][angle_index].replace('.', '').isdigit():
                new_molecule.geometry[row_index][angle_index] = str(random.uniform(0, 180))
            else:
                new_molecule.parameters[new_molecule.geometry[row_index][angle_index]] = str(random.uniform(0, 180))
    new_molecule.output = None
    new_molecule.output_values = None
    return new_molecule


def crossover_1(parent:Molecule, donor:Molecule, label:str=None) -> Molecule:
    child = parent.copy()
    child.label = label
    index = random.choice(range(2, len(child.geometry)))
    if random.choice([True, False]):
        child.geometry[index:] = donor.geometry[index:]
    else:
        child.geometry[:index] = donor.geometry[:index]
    if len(child.parameters) > 0:
        index = random.choice(range(0, len(child.parameters)))
        if random.choice([True, False]):
            child.parameters[index:] = donor.parameters[index:]
        else:
            child.parameters[:index] = donor.parameters[:index]
    child.output = None
    child.output_values = None
    return child


def crossover_2(parent:Molecule, donor:Molecule, label:str=None) -> Molecule:
    if len(parent.geometry < 5):
        raise Exception('Molecules does not has enough atoms for crossover_2.')
    child = parent.copy()
    child.label = label
    child.output = None
    child.output_values = None
    index1 = random.choice(range(2, len(child.geometry)))
    index2 = random.choice(range(2, len(child.geometry)))
    while index1 == index2 or ((index1 == 2 and index2 == len(child.geometry)) or (index1 == len(child.geometry) and index2 == 2)):
        index1 = random.choice(range(2, len(child.geometry)))
        index2 = random.choice(range(2, len(child.geometry)))
    if index1 > index2:
        index1, index2 = index2, index1
    if random.choice([True, False]):
        child.geometry[:index1] = donor.geometry[:index1]
        child.geometry[index2:] = donor.geometry[:index2]
    else:
        child.geometry[index1:index2] = donor.geometry[index1:index2]
    if len(child.parameters) > 0:
        index1 = random.choice(range(0, len(child.parameters)))
        index2 = random.choice(range(0, len(child.parameters)))
        if index1 > index2:
            index1, index2 = index2, index1
        if random.choice([True, False]):
            child.parameters[:index1] = donor.parameters[:index1]
            child.parameters[index2:] = donor.parameters[index2:]
        else:
            child.parameters[index1:index2] = donor.parameters[index1:index2]
    return child
