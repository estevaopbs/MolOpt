from __future__ import annotations
from MolOpt.molecular import Molecule
from typing import Iterable
import numpy as np
import re
import os
import cclib


"""Compatibility with Molpro  for Molecule class
"""


class Molpro_Molecule(Molecule):
    """Inherits Molecule and implements functions to be compatible with Molpro
    """

    def __init__(self, attributes: dict = dict(), elements: Iterable[str] = None,
        newcoords: np.ndarray[np.ndarray[float]] = None, distancematrix: np.ndarray[np.ndarray[float]] = None,
        connectivity: np.ndarray[int] = None, angleconnectivity: np.ndarray[int] = None,
        dihedralconnectivity: np.ndarray[int] = None, distances: np.ndarray[float] = None,
        angles: np.ndarray[float] = None, dihedrals: np.ndarray[float] = None) -> None:
        """Initializes the object

        Parameters
        ----------
        attributes : dict, optional
            Custom parameters, by default dict()
        elements : Iterable[str], optional
            Elements symbols, by default None
        newcoords : np.ndarray[np.ndarray[float]], optional
            XYZ matrix, by default None
        distancematrix : np.ndarray[np.ndarray[float]], optional
            Distance matrix, by default None
        connectivity : np.ndarray[int], optional
            Distances connectives, by default None
        angleconnectivity : np.ndarray[int], optional
            Angles connectives, by default None
        dihedralconnectivity : np.ndarray[int], optional
            Dihedrals connectives, by default None
        distances : np.ndarray[float], optional
            Distances, by default None
        angles : np.ndarray[float], optional
            Angles, by default None
        dihedrals : np.ndarray[float], optional
            Dihedrals, by default None
        """
        super().__init__(attributes, elements, newcoords, distancematrix, connectivity, angleconnectivity,
            dihedralconnectivity, distances, angles, dihedrals)
    
    def __str__(self) -> str:
        """Returns the Molpro input string

        Returns
        -------
        str
            Molpro input string
        """
        return self.inp()

    def inp(self) -> str:
        """Returns the Molpro input string

        Returns
        -------
        str
            Molpro input string
        """
        string = f"***,\n\nbasis={self.attributes['basis']}\n\n"
        if len(self.attributes['distances_parameters']) + len(self.attributes['angles_parameters'])\
            + len(self.attributes['dihedrals_parameters']) > 0:
            for parameters, values in zip((self.attributes['distances_parameters'],\
                self.attributes['angles_parameters'], self.attributes['dihedrals_parameters']),\
                    (self.distances, self.angles, self.dihedrals)): 
                if len(parameters) > 0:
                    for k, v in zip(parameters, values):
                        if k != '':
                            string += f'{k}={v}\n'
        string += '\ngeometry={' + f"{self.attributes['distance_unit']},\n"
        for n, atom in enumerate(self.elements):
            string += f'{atom}'
            if n >= 1:
                next_item = self.distances[n] if not self.attributes['distances_parameters'][n] else\
                    self.attributes['distances_parameters'][n]
                string += f", {self.connectivity[n]}, {next_item}"
            if n >= 2:
                next_item = self.angles[n] if not self.attributes['angles_parameters'][n] else\
                    self.attributes['angles_parameters'][n]
                string += f', {self.angleconnectivity[n]}, {next_item}'
            if n >= 3:
                next_item = self.dihedrals[n] if not self.attributes['dihedrals_parameters'][n] else\
                    self.attributes['dihedrals_parameters'][n]
                string += f', {self.dihedralconnectivity[n]}, {next_item}'
            string += '\n'
        string += '}\n\n' + ''.join([setting + '\n' for setting in self.attributes['settings']]) + '\n---'
        return string

    def save(self, filename: str = None, savedir: str = ''):
        """Saves the molecule in a .inp file

        Parameters
        ----------
        filename : str, optional
            File name, by default None. When it's None it becomes self.__hash__()
        savedir : str, optional
            Directory, relative to __main__ where the file will be saved, by default ''
        """
        if type(filename) != str or filename == '':
            if 'filename' in self.attributes and type(self.attributes['filename']) == str and self.attributes['filename'] != '':
                filename = self.attributes['filename']
            else:
                filename = str(self.__hash__())
        filename = re.sub('\..*', '', filename)
        savedir = savedir.rstrip('/')
        if savedir == '':
            with open(f'{filename}.inp', 'w') as file:
                file.write(self.inp())
        else:
            if not os.path.exists(savedir):
                os.mkdir(savedir)
            with open(f'{savedir}/{filename}.inp', 'w') as file:
                file.write(self.inp())

    def load(filename: str, directory: str = '', **kwargs) -> Molpro_Molecule:
        """Loads a Molpro_Molecule object from a .inp file

        Parameters
        ----------
        filename : str
            File name
        directory: str
            Directory address relative to __main__ in which the file is saved
        Returns
        -------
        Molpro_Molecule
            Loaded molecule
        """
        attributes = dict()
        elements = []
        directory = directory.rstrip('/')
        if directory != '' and directory is not None:
            directory += '/'
        with open(f'{directory}{filename}', 'r') as data:
                inpstr = re.search('\*\*\*,.*---', data.read(), flags=re.S)[0]
        if 'basis={' in inpstr:
            basis = re.sub('basis *= *', '', re.search('basis *= *{[^}]*}', inpstr)[0])
        else:
            basis = re.sub('basis *= *', '', re.search('basis *= *.*', inpstr)[0])
        z_matrix = [line.split(',') for line in re.search('geometry *= *{[^}]*}', inpstr)[0].\
            replace(' ', '').replace('geometry={', '').replace('}', '').split('\n') if line != '']
        settings = [line for line in re.sub(re.search('geometry *= *{[^}]*}', inpstr)[0], '',
            re.search('geometry *= *{[^}]*}.*', inpstr, flags=re.S)[0]).split('\n') if (line != '' and line != '---')]
        connectivity = np.zeros(len(z_matrix) - 1, dtype=int)
        distances = np.zeros(len(z_matrix) - 1)
        angleconnectivity = np.zeros(len(z_matrix) - 1, dtype=int)
        angles = np.zeros(len(z_matrix) - 1)
        dihedralconnectivity = np.zeros(len(z_matrix) - 1, dtype=int)
        dihedrals = np.zeros(len(z_matrix) - 1)
        distance_unit = z_matrix[0][0]
        distances_parameters = ['' for _ in range(len(z_matrix) - 1)]
        angles_parameters = ['' for _ in range(len(z_matrix) - 1)]
        dihedrals_parameters = ['' for _ in range(len(z_matrix) - 1)]
        for n, line in enumerate(z_matrix[1:]):
            elements.append(line[0])
            if n >= 1:
                connectivity[n] = int(line[1])
                if line[2].lstrip('-').replace('.', '', 1).isdigit():
                    distances[n] = float(line[2])
                else:
                    distances[n] = float(re.sub(f'{line[2]} *= *', '',re.search(f'{line[2]} *= *.*', inpstr)[0]))
                    distances_parameters[n] = line[2]
            if n >= 2:
                angleconnectivity[n] = int(line[3])
                if line[4].lstrip('-').replace('.', '', 1).isdigit():
                    angles[n] = float(line[4])
                else:
                    angles[n] = float(re.sub(f'{line[4]} *= *', '',re.search(f'{line[4]} *= *.*', inpstr)[0]))
                    angles_parameters[n] = line[4]
            if n >= 3:
                dihedralconnectivity[n] = int(line[5])
                if line[6].lstrip('-').replace('.', '', 1).isdigit():
                    dihedrals[n] = float(line[6])
                else:
                    dihedrals[n] = float(re.sub(f'{line[6]} *= *', '', re.search(f'{line[6]} *= *.*', inpstr)[0]))
                    dihedrals_parameters[n] = line[6]
        attributes.update({'distances_parameters' : distances_parameters, 'angles_parameters': angles_parameters,
            'dihedrals_parameters': dihedrals_parameters, 'basis': basis, 'distance_unit': distance_unit,
            'settings': settings, 'filename': filename})
        attributes.update(kwargs)
        molecule = Molpro_Molecule(attributes)
        molecule.elements = elements
        molecule.connectivity = connectivity
        molecule.angleconnectivity = angleconnectivity
        molecule.dihedralconnectivity = dihedralconnectivity
        molecule.distances = distances
        molecule.angles = angles
        molecule.dihedrals = dihedrals
        molecule.build_xyz()
        return molecule

    def print_inp(self):
        """Prints the Molpro input string
        """
        print(self.inp())

    def get_molpro_output(self, filename: str = None, savedir: str = '', save_output: bool = False, nthreads: int = 1,
        update_self: bool = True) -> cclib.parser.data.ccData_optdone_bool:
        """Parses the data from the Mopro .out or .xml output file

        Parameters
        ----------
        filename : str, optional
            The name the Molpro input will be saved, by default None
        savedir : str, optional
            Directory address relative to __main__ in which file is saved, by default ''
        save_output : bool, optional
            If it's True, the outputs will stay saved, by default False
        nthreads : int, optional
            Number of threads used in the Molpro calculation, by default 1
        update_self : bool, optional
            If its True, self.attributes will be updated with the key 'data' which value will be the data container
            object, by default True

        Returns
        -------
        cclib.parser.data.ccData_optdone_bool
            Parsed data container

        Raises
        ------
        Exception
            No convergence
        KeyError
            scfvalues not found
        """
        if type(filename)!= str or filename == '':
            if 'filename' in self.attributes and type(self.attributes['filename']) == str\
                and self.attributes['filename'] != '':
                filename = self.attributes['filename']
            else:
                filename = str(self.__hash__())
        filename = re.sub('\..*', '', filename)
        deldoc = False
        savedir = savedir.rstrip('/')
        savedir = savedir.lstrip('/')
        if savedir != '' and isinstance(savedir, str):
            savedir += '/'
        if not os.path.exists(f'{savedir}{filename}.out'):
            if not os.path.exists(f'{savedir}{filename}.inp'):
                self.save(filename, savedir)
                deldoc = True
            os.system(f'molpro -n {nthreads} ./{savedir}{filename}.inp')
        with open(f'{savedir}{filename}.out', 'r') as file:
            if 'No convergence' in file.read():
                raise Exception('No convergence')
        data = cclib.io.ccread(f'{savedir}{filename}.out')
        if not 'scfvalues' in data.__dict__:
            raise KeyError('scfvalues not found')
        if deldoc:
            os.remove(f'{savedir}{filename}.inp')
        if not save_output:
            os.remove(f'{savedir}{filename}.xml')
            os.remove(f'{savedir}{filename}.out')
        else:
            self.attributes['output'] = f'{savedir}{filename}.out'
        if update_self:
            self.attributes.update({'data': data})
        return data

    def optg(self, filename: str | None, directory: str = '', nthreads: int = 1,
        save_output: bool = False) -> Molpro_Molecule:
        """Optimizes the geometry of the molecue using Molpro optg method

        Parameters
        ----------
        filename : str | None, optional
            The name of the file the Molpro input will be saved
        directory : str, optional
            Directory address relative to __main__ in which file is saved, by default ''
        nthreads : int, optional
            Number of threads used in the Molpro calculation, by default 1
        save_output : bool, optional
            If it's True, the outputs will stay saved, by default False

        Returns
        -------
        Molpro_Molecule
            Optimized molecule
        """
        directory = directory.rstrip('/')
        if type(directory) == str and len(directory) > 0:
            directory += '/'
        opt_molecule = self.copy()
        if filename is None or type(filename) != str or filename == '':
            if 'filename' in self.attributes:
                if type(self.attributes['filename']) == str and self.attributes['filename'] != '':
                    filename = self.attributes['filename']
                else:
                    filename = str(self.__hash__())
        filename = re.sub('\..*', '', filename)
        filename = filename.replace('_optg', '')
        if not 'optg' in opt_molecule.attributes['settings']:
            opt_molecule.attributes['settings'].append('optg')
        opt_molecule.get_molpro_output(filename, directory, True, nthreads, True)
        with open(f"{directory}{filename}.out", 'r') as file:
            outstr = file.read()
        geometry = [[y for y in x.split(' ') if y != ''] for x in\
            re.findall('Current geometry \(xyz[0-9\.\nA-Za-z \(\)\-,/=]*', outstr, flags=re.S)[0].split('\n')[4:-2]]
        opt_molecule.elements = [i[0] for i in geometry]
        opt_molecule.newcoords = np.array([np.array([float(i[1]), float(i[2]), float(i[3])]) for i in geometry])
        opt_molecule.build_zmatrix()
        opt_molecule.attributes['settings'].remove('optg')
        if not save_output:
            os.remove(f"{directory}{filename}.log")
            os.remove(f"{directory}{filename}.xml")
            os.remove(f"{directory}{filename}.out")
        else:
            opt_molecule.attributes.update({'output': f"{directory}{opt_molecule.attributes['filename']}.out",
                'was_optg': True})
        opt_molecule.build_zmatrix()
        return opt_molecule
