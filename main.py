from genetic import *
from Molecular import *
import unittest
import datetime
import random


def display(candidate, start_time):
    time_diff = datetime.datetime.now() - start_time
    print("{0}\t{1}".format((candidate.Fitness), str(time_diff)))


def get_fitness(genes, fitness_param):
    #verificar
    genes.get_value([fitness_param])
    return - float(genes.output_values[fitness_param])


class test_Optimization(unittest.TestCase):
    def test_10Nb(self):
        molecule = Molecule.load('nb_n_molpro.inp', 5)
        mutate_methods = Mutate([Mutate.swap_mutate, Mutate.mutate_angles, Mutate.mutate_distances], [1, 1, 1])
        crossover_methods = Crossover([Crossover.crossover_n, Crossover.crossover_1, Crossover.crossover_2], [1, 1, 1])
        create_methods = Create([Create.randomize, Create.mutate_first], [1, 0])
        strategies = Strategies([create_methods, mutate_methods, crossover_methods], [0, 1, 1])
        max_age = 5
        pool_size = 20
        #elit_size = 0.1
        #elitism_rate = 0.1
        max_seconds = 100800
        time_tolerance = 30
        crossover_elitism = lambda x: 1
        fitness_param = '!RKS STATE 1.1 Energy'

        self.optimize(
            molecule, fitness_param, strategies, max_age, pool_size, max_seconds, time_tolerance, 
            crossover_elitism, True
            )

    def optimize(
        self, first_molecule:Molecule, fitness_param:str, strategies, max_age:int, pool_size:int, max_seconds:float, time_tolerance:int, 
        crossover_elitism:float, mutate_after_crossover:bool=False
        ):

        start_time= datetime.datetime.now()

        for strategy in strategies.strategies:
            if type(strategy) is Mutate:
                mutate_methods = strategy
            elif type(strategy) is Create:
                create_methods = strategy
            elif type(strategy) is Crossover:
                crossover_methods = strategy

        mutate_lookup = {
            Mutate.swap_mutate: lambda p, d=0: swap_mutate(p),
            Mutate.mutate_angles: lambda p, d=0: mutate_angles(p),
            Mutate.mutate_distances: lambda p, d=0: mutate_distances(p)
        }
        mutate = lambda p: mutate_lookup[random.choices(mutate_methods.methods, mutate_methods.rate)[0]](p)
        if not mutate_after_crossover:
            crossover_lookup = {
                Crossover.crossover_n: lambda p, d: crossover_n(p, d),
                Crossover.crossover_1: lambda p, d: crossover_1(p, d),
                Crossover.crossover_2: lambda p, d: crossover_2(p, d)
            }
        else:
            crossover_lookup = {
                Crossover.crossover_n: lambda p, d: mutate(crossover_n(p, d)),
                Crossover.crossover_1: lambda p, d: mutate(crossover_1(p, d)),
                Crossover.crossover_2: lambda p, d: mutate(crossover_2(p, d))
            }
        create_lookup = {
            Create.randomize: lambda p, d=0: randomize(p),
            Create.mutate_first: lambda p, d=0: mutate(first_molecule)
            # At least one mutate method is needed for this one
        }
        strategy_lookup = {
            Strategies.Create: create_lookup,
            Strategies.Mutate: mutate_lookup,
            Strategies.Crossover: crossover_lookup
        }
        def get_child(candidates, parent_index): # dar um jeito de usar bisect ao inves de deepcopy
            parent = candidates[parent_index]
            sorted_candidates = copy.deepcopy(candidates)
            sorted_candidates.sort(reverse=True, key=lambda p: p.Fitness)
            donor = random.choices(sorted_candidates, [crossover_elitism(n) for n in reversed(range(len(sorted_candidates)))])[0]
            child = Chromosome
            child.Strategy = random.choices(strategies.strategies, strategies.rate)[0]
            child.Method = random.choices(child.Strategy.methods, child.Strategy.rate)[0]
            child.Genes = strategy_lookup[child.Strategy.strategy][child.Method](parent.Genes, donor.Genes)
            child.Fitness = get_fitness(child.Genes, fitness_param)
            return child

        def fn_generate_parent():
            parent = Chromosome
            parent.Genes = randomize(first_molecule)
            parent.Fitness = get_fitness(parent.Genes, fitness_param)
            return parent
        
        first_parent = Chromosome(first_molecule, get_fitness(first_molecule, fitness_param), None, None)
        usedStrategies = []
        for timedOut, improvement in get_improvement(get_child, first_parent, fn_generate_parent, max_age, pool_size, max_seconds):
            display(improvement, start_time)
            f = (improvement.Strategy, improvement.Method)
            usedStrategies.append(f)
            if timedOut:
                improvement.Fitness.save(f'{improvement.Fitness.__hash__}_best')
                break


if __name__ == '__main__':
    unittest.main()
