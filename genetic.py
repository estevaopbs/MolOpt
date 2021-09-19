import time
from bisect import bisect_left
import random
from math import exp


class Chromosome:
    Genes = None
    Fitness = None
    Age = 0
    Strategy = None
    Method = None

    def __init__(self, genes, fitness, strategy, method):
        self.Genes = genes
        self.Fitness = fitness
        self.Strategy = strategy
        self.Method = method


class Strategies:
    Create = 0
    Mutate = 1
    Crossover = 2

    def __init__(self, strategies:list, strategies_rate:list):
        self.strategies = strategies
        self.rate = strategies_rate


class Create:
    randomize = 0
    mutate_first = 1
    strategy = 0

    def __init__(self, methods:list, methods_rate:list):
        self.methods = methods
        self.rate = methods_rate


class Mutate:
    swap_mutate = 0
    mutate_angles = 1
    mutate_distances = 2
    strategy = 1

    def __init__(self, methods:list, methods_rate:list):
        self.methods = methods
        self.rate = methods_rate


class Crossover:
    crossover_n = 0
    crossover_1 = 1
    crossover_2 = 2
    strategy = 2

    def __init__(self, methods:list, methods_rate:list):
        self.methods = methods
        self.rate = methods_rate

def get_improvement(new_child, first_parent, generate_parent, maxAge, poolSize, maxSeconds):
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
            #print(historicalFitnesses.sort(reverse=True))
            #print(bestParent.Fitness)
            yield True, bestParent
        pindex = pindex - 1 if pindex > 0 else lastParentIndex
        parent = parents[pindex]
        child = new_child(parents, pindex)
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
