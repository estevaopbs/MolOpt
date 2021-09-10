from genetic import *
from Molecular import *
import unittest
import datetime
from bisect import bisect_left
from math import exp
import random
import time


def display(candidate, start_time):
    time_diff = datetime.datetime.now() - start_time
    print("{0}\t{1}".format(candidate.Fitness), str(time_diff))


def get_fitness(genes):
    #verificar
    Fitness =  - float(genes.get_value(['!RHF STATE 1.1 Energy']))
    return Fitness

def get_crossover_rate(candidates):
    pass


class test_Optimization(unittest.TestCase):
    def test_h2co(self):
        molecule = random_molecule('H2O', 'vdz', ['hf'], 2)
        mutate_methods = Mutate([Mutate.swap_mutate, Mutate.mutate_angles, Mutate.mutate_distances], [1, 1, 1])
        crossover_methods = Crossover([Crossover.crossover_n, Crossover.crossover_1, Crossover.crossover_2], [1, 1, 1])
        create_methods = Create([Create.randomize, Create.mutate_first], [1, 0])
        strategies = Strategies([create_methods, mutate_methods, crossover_methods], [0, 1, 1])
        max_age = 20
        pool_size = 20
        #elit_size = 0.1
        #elitism_rate = 0.1
        max_seconds = 120
        generations_tolerance = 30
        crossover_elitism = lambda x: 1

        self.optimize(
            molecule, strategies, max_age, pool_size, max_seconds, generations_tolerance, 
            crossover_elitism, True
            )

    def optimize(
        self, first_molecule:Molecule, strategies, max_age:int, pool_size:int, max_seconds:float, time_tolerance:int, 
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
        def get_child(candidates, parent_index): # candidates precisa estar organizado com fitness decrescente
            parent = candidates[parent_index]
            donor = random.choices(candidates, [crossover_elitism(n) for n in reversed(range(len(candidates)))])[0]
            child = Chromosome
            child.Strategy = random.choices(strategies.strategies, strategies.rate)[0]
            child.Method = random.choices(child.Strategy.methods, child.Strategy.rate)[0]
            child.Genes = strategy_lookup[child.Strategy.strategy][child.Method](parent.Genes, donor.Genes)
            child.Fitness = get_fitness(child.Genes)
            return child

        def fn_generate_parent():
            parent = Chromosome
            parent.Genes = randomize(first_molecule)
            parent.Fitness = get_fitness(parent.Genes)
        
        first_parent = Chromosome(first_molecule, get_fitness(first_molecule), None, None)
        usedStrategies = []
        for timedOut, improvement in _get_improvement(get_child, first_parent, fn_generate_parent, max_age, pool_size, max_seconds):
            display(improvement, start_time)
            f = strategy_lookup[(improvement.Strategy, improvement.Method)]
            usedStrategies.append(f)
            if timedOut:
                best = improvement
                break
        
        best.Genes.save()

def _get_improvement(new_child, first_parent, generate_parent, maxAge, poolSize, maxSeconds):
    startTime = time.time()
    bestParent = first_parent
    yield maxSeconds is not None and time.time() - startTime > maxSeconds, bestParent
    parents = [bestParent]
    historicalFitnesses = [bestParent.Fitness]
    for _ in range(poolSize - 1):
        parent = generate_parent()
        if maxSeconds is not None and time.time() - startTime > maxSeconds:
            yield True, parent
        if parent.Fitness > bestParent.Fitness:
            yield False, parent
            bestParent = parent
            historicalFitnesses.append(parent.Fitness)
        parents.append(parent)
    lastParentIndex = poolSize - 1
    pindex = 1
    while True:
        if maxSeconds is not None and time.time() - startTime > maxSeconds:
            yield True, bestParent
        pindex = pindex - 1 if pindex > 0 else lastParentIndex
        parent = parents[pindex]
        child = new_child(parent, pindex, parents)
        if parent.Fitness > child.Fitness:
            if maxAge is None:
                continue
            parent.Age += 1
            if maxAge > parent.Age:
                continue
            index = bisect_left(historicalFitnesses, child.Fitness, 0, len(historicalFitnesses))
            difference = len(historicalFitnesses) - index
            proportionSimilar = difference / len(historicalFitnesses)
            if random.random() < exp(-proportionSimilar):
                parents[pindex] = child
                continue
            parents[pindex] = bestParent
            parent.Age = 0
            continue
        if not child.Fitness > parent.Fitness:
            child.Age = parent.Age + 1
            parents[pindex] = child
            continue
        parents[pindex] = child
        parent.Age = 0
        if child.Fitness > bestParent.Fitness:
            yield False, child
            bestParent = child
            historicalFitnesses.append(child.Fitness)



if __name__ == '__main__':
    unittest.main()