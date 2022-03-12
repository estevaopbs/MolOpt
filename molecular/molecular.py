from __future__ import annotations
import random
import copy
from typing import Iterable, Tuple
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist


"""Framework to deal with molecular structures
"""


class Molecule:
    """Framework for storing and modifying molecular geometry
    """
    __slots__ = ('attributes', 'elements', 'newcoords', 'distancematrix', 'connectivity', 'angleconnectivity',
        'dihedralconnectivity', 'distances', 'angles', 'dihedrals')

    def __init__(self, attributes: dict = dict(), elements: Iterable[str] = None,
        newcoords: np.ndarray[np.ndarray[float]] = None, distancematrix: np.ndarray[np.ndarray[float]] = None,
        connectivity: np.ndarray[int] = None, angleconnectivity: np.ndarray[int] = None,
        dihedralconnectivity: np.ndarray[int] = None, distances: np.ndarray[float] = None,
        angles: np.ndarray[float] = None, dihedrals: np.ndarray[float] = None) -> None:
        """Initializes the Molecule object

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
        self.elements = elements
        self.newcoords = newcoords
        self.distancematrix = distancematrix
        self.connectivity = connectivity
        self.angleconnectivity = angleconnectivity
        self.dihedralconnectivity = dihedralconnectivity
        self.distances = distances
        self.angles = angles
        self.dihedrals = dihedrals
        self.attributes = attributes

    def print_zmat(self) -> None:      
        """Prints its Z-matrix
        """
        if len(self.elements) > 0:
            print(f'  {self.elements[0]}')
        if len(self.elements) > 1:
            print('  {} {} {:10.6f}'.format(self.elements[1], self.connectivity[1], self.distances[1]))
        if len(self.elements) > 2:
            print('  {} {} {:10.6f} {} {:10.6f}'.format(self.elements[2], self.connectivity[2], self.distances[2],
                self.angleconnectivity[2], self.angles[2]))
        if len(self.elements) > 3:
            for n in range(3, len(self.elements)):
                print('  {} {} {:10.6f} {} {:10.6f} {} {:10.6f}'.format(self.elements[n], self.connectivity[n],
                    self.distances[n], self.angleconnectivity[n], self.angles[n], self.dihedralconnectivity[n],
                    self.dihedrals[n]))

    def print_xyz(self) -> None:
        """Prints its XYZ matrix
        """
        if not self.newcoords.all():
            self.build_xyz()
        atomcoords = [x.tolist() for x in self.newcoords]
        for i in range(len(atomcoords)):
            atomcoords[i].insert(0, self.elements[i])
        for atom in atomcoords:
            print("  %s %10.6f %10.6f %10.6f" % tuple(atom))
    
    def copy(self) -> Molecule:
        """Returns a deepcopy of itself

        Returns
        -------
        Molecule
            Deepcopy of itself
        """        
        return copy.deepcopy(self)

    def build_xyz(self) -> None:
        """Builds its XYZ matrix from its Z-matrix 
        """        
        atomnames = self.elements
        rconnect = self.connectivity[1:]
        rlist = self.distances[1:]
        aconnect = self.angleconnectivity[2:]
        alist = self.angles[2:]
        dconnect = self.dihedralconnectivity[3:]
        dlist = self.dihedrals[3:]
        npart = len(atomnames)
        xyzarr = np.zeros([npart, 3])
        if (npart > 1):
            xyzarr[1] = [rlist[0], 0.0, 0.0]
        if (npart > 2):
            i = rconnect[1] - 1
            j = aconnect[0] - 1
            r = rlist[1]
            theta = alist[0] * np.pi / 180.0
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            a_i = xyzarr[i]
            b_ij = xyzarr[j] - xyzarr[i]
            if (b_ij[0] < 0):
                x = a_i[0] - x
                y = a_i[1] - y
            else:
                x = a_i[0] + x
                y = a_i[1] + y
            xyzarr[2] = [x, y, 0.0]
        for n in range(3, npart):
            r = rlist[n-1]
            theta = alist[n-2] * np.pi / 180.0
            phi = dlist[n-3] * np.pi / 180.0
            sinTheta = np.sin(theta)
            cosTheta = np.cos(theta)
            sinPhi = np.sin(phi)
            cosPhi = np.cos(phi)
            x = r * cosTheta
            y = r * cosPhi * sinTheta
            z = r * sinPhi * sinTheta
            i = rconnect[n-1] - 1
            j = aconnect[n-2] - 1
            k = dconnect[n-3] - 1
            a = xyzarr[k]
            b = xyzarr[j]
            c = xyzarr[i]
            ab = b - a
            bc = c - b
            bc = bc / np.linalg.norm(bc)
            nv = np.cross(ab, bc)
            nv = nv / np.linalg.norm(nv)
            ncbc = np.cross(nv, bc)
            new_x = c[0] - bc[0] * x + ncbc[0] * y + nv[0] * z
            new_y = c[1] - bc[1] * x + ncbc[1] * y + nv[1] * z
            new_z = c[2] - bc[2] * x + ncbc[2] * y + nv[2] * z
            xyzarr[n] = [new_x, new_y, new_z]
        self.newcoords = xyzarr

    def build_zmatrix(self):
        """Builds its Z-matrix from its XYZ matrix
        """        
        xyzarr = self.newcoords
        distmat = distance_matrix(self.newcoords)
        rlist = []
        npart, ncoord = xyzarr.shape
        alist = []
        dlist = []
        rcon = []
        acon = []
        dcon = []
        if npart > 0:
            rcon.append(0)
            acon.append(0)
            dcon.append(0)
            rlist.append(0)
            alist.append(0)
            dlist.append(0)
            if npart > 1:
                rcon.append(1)
                acon.append(0)
                dcon.append(0)
                alist.append(0)
                dlist.append(0)
                rlist.append(distmat[0][1])
                if npart > 2:
                    rcon.append(1)
                    acon.append(2)
                    dcon.append(0)
                    dlist.append(0)
                    rlist.append(distmat[0][2])
                    alist.append(angle(xyzarr, 2, 0, 1))
                    if npart > 3:
                        for i in range(3, npart):
                            rcon.append(i - 2)
                            acon.append(i - 1)
                            dcon.append(i)
                            rlist.append(distmat[i-3][i])
                            alist.append(angle(xyzarr, i, i-3, i-2))
                            dlist.append(dihedral(xyzarr, i, i-3, i-2, i-1))
        self.distances = np.array(rlist)
        self.connectivity = np.array(rcon)
        self.angles = np.array(alist)
        self.angleconnectivity = np.array(acon)
        self.dihedrals = np.array(dlist)
        self.dihedralconnectivity = np.array(dcon)


def angle(xyzarr: np.ndarray[np.ndarray[float,float,float]], i: int, j: int, k: int) -> float:
    """Returns the angle formed by three points

    Parameters
    ----------
    xyzarr : np.ndarray[np.ndarray[float,float,float]]
        XYZ matrix
    i : int
        Index of the first point in xyzarr
    j : int
        Index of the second point in xyzarr
    k : int
        Index of the third point in xyzarr

    Returns
    -------
    float
        Angle
    """    
    rij = xyzarr[i] - xyzarr[j]
    rkj = xyzarr[k] - xyzarr[j]
    cos_theta = np.dot(rij, rkj)
    sin_theta = np.linalg.norm(np.cross(rij, rkj))
    theta = np.arctan2(sin_theta, cos_theta)
    theta = 180.0 * theta / np.pi
    return theta


def dihedral(xyzarr, i, j, k, l) -> float:
    """Returns the dihedral angle formed by four points

    Parameters
    ----------
    xyzarr : _type_
        XYZ matrix
    i : _type_
        Index of the first point in xyzarr
    j : _type_
        Index of the second point in xyzarr
    k : _type_
        Index of the third point in xyzarr
    l : _type_
        Index of the fourth point in xyzarr

    Returns
    -------
    float
        Dihedral angle
    """    
    rji = xyzarr[j] - xyzarr[i]
    rkj = xyzarr[k] - xyzarr[j]
    rlk = xyzarr[l] - xyzarr[k]
    v1 = np.cross(rji, rkj)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(rlk, rkj)
    v2 = v2 / np.linalg.norm(v2)
    m1 = np.cross(v1, rkj) / np.linalg.norm(rkj)
    x = np.dot(v1, v2)
    y = np.dot(m1, v2)
    chi = np.arctan2(y, x)
    chi = -180.0 - 180.0 * chi / np.pi
    if (chi < -180.0):
        chi = chi + 360.0
    return chi


def distance_matrix(xyzarr: np.ndarray) -> np.ndarray:
    """Receives a XYZ matrix and returns its distance matrix

    Parameters
    ----------
    xyzarr : np.ndarray
        XYZ matrix

    Returns
    -------
    np.ndarray
        Distance matrix
    """    
    return cdist(xyzarr, xyzarr)


def randomize(molecule: Molecule, dist_range: Tuple[float, float]) -> Molecule:
    """Returns the molecule with randomized atoms coordinates

    Parameters
    ----------
    molecule : Molecule
        Molecule which coordinates will be randomized  
    dist_range : Tuple[float, float]
        Range of the distances that will be randomly generated beteween the atoms in the Z-matrix

    Returns
    -------
    Molecule
        Randomized molecule
    """    
    new_molecule = molecule.copy()
    for i in range(1, len(molecule.elements)):
        new_molecule.distances[i] = random.uniform(dist_range[0], dist_range[1])
        if i >= 2:
            new_molecule.angles[i] = random.uniform(0, 360)
        if i >= 3:
            new_molecule.dihedrals[i] = random.uniform(0, 360)
    new_molecule.build_xyz()
    return new_molecule


def piece_rotation(molecule: Molecule, indexes: Iterable[int], rotation_origin: np.array[float, float, float],
    angles: np.array[float, float, float]) -> Molecule:
    """Produces a rotation over a piece of the molecule

    Parameters
    ----------
    molecule : Molecule
        Molecule which a piece will be rotated
    indexes : Iterable[int]
        Indexes of the particles, in the XYZ matrix, that compose the piece
    rotation_origin : np.array[float, float, float]
        Point arround which the atoms will be rotated
    angles : np.array[float, float, float]
        Angles in which the piece will be rotated in the x, y and z axis consecutively

    Returns
    -------
    Molecule
        New molecule
    """    
    new_molecule = molecule.copy()
    rotation = Rotation.from_euler('xyz', angles, degrees=True)
    for i in indexes:
        new_molecule.newcoords[i] = rotation.apply(new_molecule.newcoords[i] - rotation_origin) + rotation_origin
    new_molecule.build_zmatrix()
    return new_molecule


def piece_displacement(molecule: Molecule, indexes: Iterable[int] | int,
    displacement_vector: np.array[float, float, float]) -> Molecule:
    """Produces a displacement over a piece of the molecule

    Parameters
    ----------
    molecule : Molecule
        Molecule which a piece will be displaced
    indexes : Iterable[int] | int
        Indexes of the particles, in the XYZ matrix, that compose the piece
    displacement_vector : np.array[float, float, float]
        Vector in which all the atoms of the piece will be displaced

    Returns
    -------
    Molecule
        New molecule
    """    
    new_molecule = molecule.copy()
    for i in indexes:
        new_molecule.newcoords[i] += displacement_vector
    new_molecule.build_zmatrix()
    return new_molecule


def particle_permutation(molecule: Molecule, indexes: Tuple[int, int]) -> Molecule:
    """Permutates the position of two atoms

    Parameters
    ----------
    molecule : Molecule
        Molecule in which two atoms will permutate
    indexes : Tuple[int, int]
        Atoms indexes

    Returns
    -------
    Molecule
        New molecule
    """    
    new_molecule = molecule.copy()
    new_molecule.elements[indexes[0]], new_molecule.elements[indexes[1]] =\
        new_molecule.elements[indexes[1]], new_molecule.elements[indexes[0]]
    return new_molecule


def piece_reflection(molecule: Molecule, indexes: Iterable[int],
    reflection_plane_normal_vector: np.ndarray[float, float, float],
    reflection_plane_point: np.ndarray[float, float, float], replace_original: bool,
    image_displacement_vector: np.ndarray[float, float, float] = np.zeros(3),
    replacing_indexes: Iterable[int] = None) -> Molecule:
    """Produces a reflection in a piece of the molecule

    Parameters
    ----------
    molecule : Molecule
        Molecule which piece will be reflected
    indexes : Iterable[int]
        Indexes of the atoms, in the XYZ matrix, that compose the piece
    reflection_plane_normal_vector : np.ndarray[float, float, float]
        Normal vector of the reflection plane
    reflection_plane_point : np.ndarray[float, float, float]
        A point which is in the reflection plane
    replace_original : bool
        If True, the coordinates of the reflected piece will override the coordinates of the original one in the XYZ
        matrix
    image_displacement_vector : np.ndarray[float, float, float], optional
        The vector by which the reflected piece will be displaced, by default np.zeros(3)
    replacing_indexes : Iterable[int], optional
        The indexes of the atoms the reflected piece will override, by default None

    Returns
    -------
    Molecule
        New molecule
    """
    new_molecule = molecule.copy()
    mirror_image = np.zeros((len(indexes), 3))
    for n, i in enumerate(indexes):
        mirror_image[n] = 2 * (np.dot(reflection_plane_normal_vector,
            (reflection_plane_point - new_molecule.newcoords[i]))/np.dot(reflection_plane_normal_vector,
                reflection_plane_normal_vector)) * reflection_plane_normal_vector + new_molecule.newcoords[i]
    if replace_original:
        for n, i in enumerate(indexes):
            new_molecule.newcoords[i] = mirror_image[n] + image_displacement_vector
    if replacing_indexes is not None:
        for n, i in enumerate(replacing_indexes):
            new_molecule.newcoords[i] = mirror_image[n] + image_displacement_vector
    new_molecule.build_zmatrix()
    return new_molecule


def enlarge_reduce(molecule: Molecule, rate: float) -> Molecule:
    """Multiplies the distances between all atoms by a value

    Parameters
    ----------
    molecule : Molecule
        Molecule which will be enlarged or reduced
    rate : float
        The value by which all the distances between the atoms will be multiplied

    Returns
    -------
    Molecule
        New molecule
    """
    new_molecule = molecule.copy()
    new_molecule.distances *= rate
    new_molecule.build_xyz()
    return new_molecule
