from genetic import *
from Molecular import *
import unittest
import datetime
import time
from bisect import bisect_left
from math import exp
import random


def display(candidate, start_time):
    time_diff = datetime.datetime.now() - start_time
    print("{0}\t{1}\t{2}".format(candidate.total_energy), str(time_diff))


def get_fitness(candidate):
    #verificar
    candidate.Fitness =  - float(candidate.get_value('total energy'))
    return candidate.Fitness


class test_Optimization(unittest.TestCase):
    def test_h2co(self):
        file = None
        mutate_methods = [
            Mutate.swap_mutate, 
            Mutate.mutate_distances, 
            Mutate.mutate_angles
            ]
        methods_rate = [0.2, 0.4, 0.4]
        crossover_methods = [crossover_n]
        crossover_methods_rate = [1]
        atoms = 'H2O'
        basis = 'vdz'
        settings = ['SET,CHARGE=0', 'hf', 'CCSD(T)', 'optg', 'freq']
        rand_range = 2
        dist_unit = 'ang'

        max_age = 20
        pool_size = 20
        elitism_rate = 0.2
        crossover_rate = 0.2
        max_seconds = 120

        self.optmize(file, mutate_methods, atoms, basis, settings, rand_range, dist_unit, max_age, pool_size, 
                     elitism_rate, max_seconds, crossover_rate, methods_rate)

    def optimize(file, mutate_methods, atoms=None, basis=None, settings=None, rand_range=None, dist_unit=None,
                 max_age=20, pool_size=1, elitism_rate=0, max_seconds=60, crossover_rate=0, mutate_methods_rate=None,
                 crossover_methods = [crossover_n], crossover_methods_rate=[1], crossover_elitism=lambda x: 1,
                 create_methods=None):

        start_time= datetime.datetime.now()

        strategy_lookup = {
        Strategies.Create: lambda: create_candidate(),
        Strategies.Mutate: Mutate,
        Strategies.Crossover: Crossover
        }
        mutate_lookup = {
            Mutate.swap_mutate: lambda x: Molecular.swap_mutate(x),
            Mutate.mutate_angles: lambda x: Molecular.mutate_angles(x),
            Mutate.mutate_distances: lambda x: Molecular.mutate_distances(x)
        }
        crossover_lookup = {
            Crossover.crossover_n: lambda p, d: Molecular.n_crossover(p, d)
        }

        if mutate_methods_rate is None:
            mutate_methods_rate = [1 for _ in mutate_methods]

        def mutate(candidate):
            mutate_method = random.choices(mutate_methods, mutate_methods_rate)[0]
            return mutate_method(candidate)

        def crossover(candidates, parent_index):
            crossover_weights = [crossover_elitism(n) for n in [i + 1 for i in reversed(range(len(candidates)))]]
            donor = random.choices(candidates, crossover_weights)
            return random.choices(crossover_methods, crossover_methods_rate)(candidates[parent_index], donor)

        def get_child(candidates, parent_index):
            parent = candidates[parent_index]
            if random.choices([0, 1], [1 - crossover_rate, crossover_rate])[0] == 0:
                child_molecule = candidates[parent_index].copy()
                child_molecule, strategy = mutate(child_molecule)
            else:
                child_molecule = crossover(candidates, parent_index)
            return Chromosome(child_molecule, get_fitness(child_molecule), )

        if file is not None:
            first_molecule = Molecule.load(file)

            def create_candidate():
                candidate_molecule = mutate(first_molecule.copy())
                return Chromosome(candidate_molecule, get_fitness(candidate_molecule), Strategies.Create)

        else:            
            def create_candidate():
                return random_molecule(atoms, basis, settings, rand_range, dist_unit)
        
        def fn_display(candidate):
            display(candidate, start_time)

        

        #best = get_best(get_child, create_candidate, max_age, pool_size, max_seconds)
        #best.save()


if __name__ == '__main__':
    unittest.main()