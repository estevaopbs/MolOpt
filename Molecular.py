import re
import random
import os


class Molecule:
    __slots__ = ('basis', 'parameters', 'geometry', 'settings', 'rand_range', 'label', 'total_energy', 
                 'output_values', 'mutate_methods', 'output')
    def __init__(self, basis:str, geometry:list, settings:list, parameters:dict=dict(), rand_range=None, label:str=None, 
                 total_energy=None, output:str=None, output_values:dict=dict(), mutate_methods:list=None):
        self.basis = basis
        self.parameters = parameters
        self.geometry = geometry
        self.settings = settings
        self.rand_range = rand_range
        self.label = label
        self.total_energy = total_energy
        self.output = output
        self.output_values = output_values
        self.mutate_methods = mutate_methods

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
        document = document if document is not None else self.label if self.label is not None else self.__hash__()
        directory += '/'
        if not os.path.exists(directory):
            os.mkdir(directory)
        open(f'{directory}/{document}.inp', 'w').write(str(self))

    def get_value(self, wanted:list, document=None, directory:str='data',wait:bool=True) -> None:
        document = document if document is not None else self.label if self.label is not None else self.__hash__()
        directory += '/'
        output_address = f'{directory}{document[:-3]}out'
        if not os.path.isfile(f'{output_address}'):
            os.system(f'molpro {directory}{document}')
        if wait or os.path.isfile(f'{output_address}'):
            while not os.path.isfile(f'{output_address}'):
                continue
            with open(f'{directory}{document[:-3]}out', 'r') as output:
                outstr = output.read()
            for item in wanted:
                self.output_values.update({item: re.search(f'{item}.*', outstr)[0]})
        self.output = f'{document[:-3]}out'

    @staticmethod
    def load(file:str, rand_range=None, label:str=None, total_energy=None, output=None, output_values=None,
             mutate_methods=None):
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
        return Molecule(basis, geometry, settings, parameters, rand_range, label, total_energy, output, output_values, 
                        mutate_methods)
    
    def swap_mutate(self):
        index0 = random.choice(range(1, len(self.geometry)))
        index1 = random.choice(range(2, len(self.geometry[index0]), 2))
        if index1 == 0:
            index2 = random.choice(range(1, len(self.geometry)))
            index3 = 0
        elif index1 == 2:
            index2 = random.choice(range(1, len(self.geometry)))
            index3 = 2
        else:
            index2 = random.choice(range(3, len(self.geometry)))
            index3 = random.choice(range(4, len(self.geometry[index0]), 2))
        self.geometry[index0][index1], self.geometry[index2][index3] =\
        self.geometry[index2][index3], self.geometry[index0][index1]
    
    def mutate_angles(self):
        index0 = random.choice(range(3, len(self.geometry)))
        index1 = random.choice(range(4, len(self.geometry[index0]), 2))
        if self.geometry[index0][index1].replace('.', '').isdigit():
            self.geometry[index0][index1] = str(random.uniform(0, 180))
        else:
            self.parameters[self.geometry[index0][index1]] = str(random.uniform(0, 180))
    
    def mutate_distances(self):
        index0 = random.choice(range(1, len(self.geometry)))
        if self.geometry[index0][2].replace('.', '').isdigit():
            self.geometry[index0][2] = str(random.uniform(0, self.rand_range))
        else:
            self.parameters[self.geometry[index0][2]] = str(random.uniform(0, self.rand_range))
    
    def mutate(self):
        random.choice(self.mutate_methods)(self)

    def copy(self):
        return(Molecule(self.basis, self.parameters, self.geometry, self.settings, self.rand_range, self.label, 
                        self.total_energy, self.output, self.output_values, self.mutate_methods))


def random_molecule(molecule:str, basis:str, settings:list, rand_range:float, label:str=None, dist_unit:str='ang') -> Molecule:
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


def n_crossover(parent:Molecule, donor:Molecule, label:str=None) -> Molecule:
    child = parent.copy()
    child.label = label
    n_parameters = random.randint(1, len(child.parameters)-2)
    n_atoms = random.randint(1, len(child.geometry) - 2)
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
        index = random.randint(1, len(child.geometry))
        while index in atoms_indexes:
            index = random.randint(1, len(child.geometry))
        atoms_indexes.append(index)
        child.parameters[index] = donor.parameters[index]
    return child
