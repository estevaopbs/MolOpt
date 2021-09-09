from genetic import *
from Molecular import *
import unittest
import datetime
from bisect import bisect_left
from math import exp
import random


def display(candidate, start_time):
    time_diff = datetime.datetime.now() - start_time
    print("{0}\t{1}".format(candidate.total_energy), str(time_diff))


def get_fitness(candidate):
    #verificar
    candidate.Fitness =  - float(candidate.Genes.get_value('total energy'))
    return candidate.Fitness


class test_Optimization(unittest.TestCase):
    def test_h2co(self):
        molecule = random_molecule('H2O', 'vdz', ['hf'], 2)
        mutate_methods = Mutate([Mutate.swap_mutate, Mutate.mutate_angles, Mutate.mutate_distances], [1, 1, 1])
        crossover_methods = Crossover([Crossover.crossover_n, Crossover.crossover_1, Crossover.crossover_2], [1, 1, 1])
        create_methods = Create([Create.randomize, Create.mutate_first], [1, 0])
        strategies = Strategies([create_methods, mutate_methods, crossover_methods], [0, 1, 1])
        max_age = 20
        pool_size = 20
        elit_size = 0.1
        elitism_rate = 0.2 
        max_seconds = 120
        generations_tolerane = 20
        crossover_elitism = lambda x: 1

        self.optimize(
            molecule, strategies, max_age, pool_size, elit_size, elitism_rate, max_seconds, generations_tolerane, 
            crossover_elitism
            )

    def optimize(
        self, first_molecule:Molecule, strategies, max_age:int, pool_size:int, elit_size:float, elitism_rate:float, 
        max_seconds:float, generations_tolerance:int, crossover_elitism
        ):

        start_time= datetime.datetime.now()

        for strategy in strategies.strategies:
            if type(strategy) is Mutate:
                mutate_methods = strategy
                break

        mutate_lookup = {
            Mutate.swap_mutate: lambda p, d=0: swap_mutate(p),
            Mutate.mutate_angles: lambda p, d=0: mutate_angles(p),
            Mutate.mutate_distances: lambda p, d=0: mutate_distances(p)
        }
        crossover_lookup = {
            Crossover.crossover_n: lambda p, d: crossover_n(p, d),
            Crossover.crossover_1: lambda p, d: crossover_1(p, d),
            Crossover.crossover_2: lambda p, d: crossover_2(p, d)
        }
        create_lookup = {
            Create.randomize: lambda p, d=0: randomize(p),
            Create.mutate_first: lambda p, d=0: mutate_lookup[random.choices(mutate_methods.methods, mutate_methods.rate)[0]](first_molecule)
            # At least one mutate method is needed for this one
        }
        strategy_lookup = {
            Strategies.Create: create_lookup,
            Strategies.Mutate: mutate_lookup,
            Strategies.Crossover: crossover_lookup
        }

        def get_child(candidates, parent_index): # candidates precisa estar organizado com fitness decrescente
            parent = candidates[parent_index]
            donor = random.choices(candidates, [crossover_elitism(n) for n in reversed(range(len(candidates)))])[0]
            child = Chromosome
            child.Strategy = random.choices(strategies.strategies, strategies.rate)[0]
            child.Method = random.choices(child.Strategy.methods, child.Strategy.rate)[0]
            child.Genes = strategy_lookup[child.Strategy.strategy][child.Method](parent.Genes, donor.Genes)
            get_fitness(child)
            return child
        
        def fn_display(candidate):
            display(candidate, start_time)


if __name__ == '__main__':
    unittest.main()