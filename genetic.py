import time
from bisect import bisect_left
from math import exp
import multiprocessing as mp
from typing import Union, Any
from abc import ABC, abstractmethod
import random
import copy
import os

"""Problem-specific genetic algorithm to deal with molecular geometry optimization using Molpro and molecular module
molecular
"""

class Chromosome:
    """Object that represents the candidates
    """    
    Chromosome = Any 
    __slots__ = ('genes', 'fitness', 'strategy', 'age', 'lineage', 'label')

    def __init__(self, genes: Any = None, fitness: Union[float, int] = None, strategy: list = [], 
        age: int = 0, lineage: list[Chromosome] = [], label: str = None): 
        self.genes = genes
        self.fitness = fitness
        self.strategy = strategy
        self.age = age
        self.lineage = lineage
        self.label = label

    @property
    def strategy_str(self): 
        return str([strategy.__name__ for strategy in self.strategy]).replace("'", "")


class Strategies: 
    __slots__ = ('strategies', 'rate')

    def __init__(self, strategies: list, strategies_rate: list):  
        self.strategies = strategies
        self.rate = strategies_rate


class Create:
    __slots__ = ('methods', 'rate')

    def __init__(self, methods:list, methods_rate:list):
        self.methods = methods
        self.rate = methods_rate

    def __call__(self, parent=None, donor=None, mutate_after_crossover=None, mutate_methods=None, first_parent=None):
        child = Chromosome()
        method = random.choices(self.methods, self.rate)[0]
        child.genes = method(first_parent)
        child.strategy.append(method)
        return child


class Mutate:
    __slots__ = ('methods', 'rate')
    
    def __init__(self, methods:list, methods_rate:list):
        self.methods = methods
        self.rate = methods_rate

    def __call__(self, parent, donor=None, mutate_after_crossover=None, mutate_methods=None, first_parent=None):
        child = Chromosome()
        method = random.choices(self.methods, self.rate)[0]
        child.genes = method(parent)
        child.strategy.append(method)
        return child


class Crossover:
    __slots__ = ('methods', 'rate')

    def __init__(self, methods:list, methods_rate:list):
        self.methods = methods
        self.rate = methods_rate

    def __call__(self, parent, donor, mutate_after_crossover, mutate_methods, first_parent=None):
        child = Chromosome()
        method = random.choices(self.methods, self.rate)[0]
        child.lineage += donor.lineage
        child.lineage.append(donor)
        child.genes = method(parent, donor)
        child.strategy.append(method)
        if mutate_after_crossover:
            return mutate_methods(child)
        return child


def mutate_first():
    pass


class Genetic(ABC):
    def __init__(self, first_genes, fitness_param, strategies, max_age, pool_size, mutate_after_crossover, 
        crossover_elitism, elitism_rate, freedom_rate, parallelism, local_opt, max_seconds, time_toler, gens_toler, 
        max_gens):
        self.first_genes = first_genes
        self.fitness_param = fitness_param
        self.strategies = strategies
        self.max_age = max_age
        self.pool_size = pool_size
        self.mutate_after_crossover = mutate_after_crossover
        self.crossover_elitism = [1 for _ in range(pool_size)] if crossover_elitism is None else crossover_elitism
        self.elitism_rate = elitism_rate
        self.freedom_rate = freedom_rate
        self.parallelism = parallelism
        self.local_opt = local_opt
        self.max_seconds = max_seconds
        self.time_toler = time_toler
        self.gens_toler = gens_toler
        self.max_gens = max_gens
        self.start_time = None
        self.first_parent = None
        self.lineage_ids = []
        for strategy in strategies.strategies:
            if type(strategy) is Mutate:
                self.mutate_methods = strategy
            elif type(strategy) is Create:
                self.create_methods = strategy
            elif type(strategy) is Crossover:
                self.crossover_methods = strategy
        if mutate_first in self.create_methods.methods:
            self.create_methods.methods[self.create_methods.methods.index(mutate_first)] = self.mutate_first
        
    @abstractmethod
    def get_fitness(self, candidate):
        pass

    @abstractmethod
    def save(self, candidate, file_name, directory):
        pass

    def mutate_first(self, first_parent):
        return self.mutate_methods(first_parent).genes

    @staticmethod
    def catch(candidate):
        raise  Exception(f'An exception ocurred while generating candidate {candidate.label}.')
    
    @staticmethod
    def display(candidate, timediff):
        print("{0}\t{1}".format((candidate.fitness), str(timediff)))
       
    def load(self) -> Chromosome:
        return Chromosome(self.first_genes, self.first_parent.fitness, [self.load])

    def __get_child(self, candidates, parent_index, queue:mp.Queue=None, child_index:int=None, label:str=None):
        sorted_candidates = copy.copy(candidates)
        sorted_candidates.sort(reverse=True, key=lambda p: p.fitness)
        while True:
            try:
                parent = candidates[parent_index]
                for _ in range(self.freedom_rate):
                    donor = random.choices(sorted_candidates, self.crossover_elitism)[0]
                    child = random.choices(self.strategies.strategies, self.strategies.rate)[0]\
                        (parent, donor, self.mutate_after_crossover, self.mutate_methods, self.first_parent)
                    parent = child
                child.label = label
                child.fitness = self.get_fitness(child)
                child.lineage += parent.lineage + [parent]
                child.lineage = list(set(child.lineage))
                child.lineage = [ancestor for ancestor in child.lineage if ancestor.label not in self.lineage_ids]
                break
            except:
                self.catch(child)
        if queue is not None:
            queue.put({child_index: child})
        return child

    def __local_optimization(self, candidate:Chromosome) -> Chromosome:
        if not self.local_opt:
            return candidate
        new_genes = self.local_optimize(candidate)
        new_fitness = self.get_fitness(candidate)
        return Chromosome(new_genes, new_fitness, candidate.strategy + [self.local_optimize], 0, 
            candidate.lineage + [candidate])

    def local_optimize(self, candidate):
        pass

    def __generate_parent(self, queue=None, label:str=None):
        while True:
            try:
                parent = self.create_methods(first_parent=self.first_parent)
                parent.label = label
                parent.fitness = self.get_fitness(parent)
                break
            except:
                self.catch(parent)
        if queue is not None:
            queue.put(parent)
        return parent

    def run(self):
        self.start_time = time.time()
        self.first_parent = Chromosome(self.first_genes, label = '0_0')
        self.first_parent.fitness = self.get_fitness(self.first_parent)
        opt_func = self.__get_improvement_mp if self.parallelism else self.__get_improvement
        n = 0
        j = 0
        if not os.path.exists('lineage'):
            os.mkdir('lineage')
        for timed_out, improvement in opt_func():
            self.save(improvement, f'{n}_{improvement.label}', 'improvements')
            improvement.lineage.append(improvement)
            timediff = time.time() - self.start_time
            self.display(improvement, timediff)
            with open('improvements_strategies.log', 'a') as islog:
                islog.write(f'{improvement.strategy_str}\t{improvement.fitness}\t{timediff}\n')
            with open('lineage_strategies.log', 'a') as lslog:
                for ancestor in improvement.lineage:
                    if not ancestor.label in self.lineage_ids:
                        self.lineage_ids.append(ancestor.label)
                        lslog.write(f'{ancestor.strategy_str}\t{ancestor.fitness}\t{timediff}\n')
                        self.save(ancestor, f'{j}_{ancestor.label}', 'lineage')
                        j += 1
            n += 1
            if timed_out:
                break
        improvement.genes.save(f'{j}_{ancestor.label}', 'lineage')
        with open('strategies_log.txt', 'a') as slog:
            slog.write('\n---\n\n')
        return improvement.genes

    def __get_improvement(self):
        best_parent = self.__local_optimization(self.load())
        best_parent.label = '0_0'
        yield self.max_seconds is not None and time.time() - self.start_time > self.max_seconds, best_parent
        best_parent.lineage = []
        parents = [best_parent]
        historical_fitnesses = [best_parent.fitness]
        for n in range(self.pool_size - 1):
            parent = self.__generate_parent(label=f'{n+1}_{n+1}')
            if self.max_seconds is not None and time.time() - self.start_time > self.max_seconds:
                yield True, parent
            if self.time_toler is not None and time.time() - last_improvement_time > self.time_toler:
                yield True, best_parent
            if parent.fitness > best_parent.fitness:
                parent = self.__local_optimization(parent)
                yield False, parent
                last_improvement_time = time.time()
                best_parent = parent
                best_parent.lineage = []
                historical_fitnesses.append(parent.fitness)
            parents.append(parent)
        lastParentIndex = self.pool_size - 1
        pindex = 1
        n = self.pool_size
        while True:
            n += 1
            if self.max_seconds is not None and time.time() - self.start_time > self.max_seconds:
                yield True, best_parent
            if self.time_toler is not None and time.time() - last_improvement_time > self.time_toler:
                yield True, best_parent
            pindex = pindex - 1 if pindex > 0 else lastParentIndex
            parent = parents[pindex]
            child = self.__get_child(parents, pindex, label=f'{n}_{pindex}')
            if parent.fitness > child.fitness:
                if self.max_age is None:
                    continue
                parent.age += 1
                if self.max_age > parent.age:
                    continue
                index = bisect_left(historical_fitnesses, child.fitness, 0, len(historical_fitnesses))
                difference = len(historical_fitnesses) - index
                proportionSimilar = difference / len(historical_fitnesses)
                if random.random() < exp(-proportionSimilar):
                    parents[pindex] = child
                    continue
                parents[pindex] = best_parent
                parent.age = 0
                continue
            if not child.fitness > parent.fitness:
                child.age = parent.age + 1
                parents[pindex] = child
                continue
            parents[pindex] = child
            parent.age = 0
            if child.fitness > best_parent.fitness:
                child = self.__local_optimization(child)
                yield False, child
                last_improvement_time = time.time()
                best_parent = child
                best_parent.lineage = []
                historical_fitnesses.append(child.fitness)

    def __get_improvement_mp(self):
        elit_size = len(self.elitism_rate) if self.elitism_rate is not None else None
        last_improvement_time = self.start_time
        queue = mp.Queue(maxsize=self.pool_size - 1)
        processes = []
        gen = 0
        if elit_size is not None:
            if sum(self.elitism_rate) > self.pool_size:
                raise Exception('Minimal elitism exceeds pool size. Increase the pool size or reduce the elit size.')
        best_parent = self.__local_optimization(self.load())
        yield self.max_seconds is not None and time.time() - self.start_time > self.max_seconds, best_parent
        best_parent.lineage = []
        parents = [best_parent]
        historical_fitnesses = [best_parent.fitness]
        for n in range(self.pool_size - 1):
            processes.append(mp.Process(target=self.__generate_parent, args=(queue, f'{gen}_{n+1}')))
        for process in processes:
            process.start()
        for _ in range(self.pool_size - 1):
            parents.append(queue.get())
        for process in processes:
            process.join()
        sorted_next_gen = copy.copy(parents)
        sorted_next_gen.sort(key=lambda c: c.fitness, reverse=False)
        for parent in sorted_next_gen:     
            if parent.fitness > best_parent.fitness:
                parent = self.__local_optimization(parent)
                yield False, parent
                best_parent = parent
                best_parent.lineage = []
                last_improvement_time = time.time()
                historical_fitnesses.append(parent.fitness)
        parents.sort(key=lambda p: p.fitness, reverse=True)
        last_improvement_gen = 0
        while True:
            gen += 1
            if self.max_seconds is not None and time.time() - self.start_time > self.max_seconds:
                yield True, best_parent
            if self.max_gens is not None and gen > self.max_gens:
                yield True, best_parent
            if self.gens_toler is not None and gen - last_improvement_gen > self.gens_toler + 1:
                yield True, best_parent
            if self.time_toler is not None and time.time() - last_improvement_time > self.time_toler:
                yield True, best_parent
            next_gen = []
            queue = mp.Queue(maxsize=self.pool_size)
            processes = []
            results = dict()
            if elit_size is not None:
                for pindex in range(elit_size):
                    for i in range(self.elitism_rate[pindex]):
                        processes.append(mp.Process(target=self.__get_child, args=(parents, pindex, queue, i + \
                            sum(self.elitism_rate[:pindex]), f'{gen}_{i + sum(self.elitism_rate[:pindex])}')))
                for pindex in range(elit_size, self.pool_size - sum(self.elitism_rate) + elit_size):
                    processes.append(mp.Process(target=self.__get_child, args=(parents, pindex, queue, 
                        sum(self.elitism_rate) + pindex - elit_size, 
                        f'{gen}_{sum(self.elitism_rate) + pindex - elit_size}')))
            else:
                for pindex in range(self.pool_size):
                    processes.append(mp.Process(target=self.__get_child, args=(parents, pindex, queue, pindex,
                    f'{gen}_{pindex}')))
            for process in processes:
                process.start()
            for _ in range(self.pool_size):
                results.update(queue.get())
            for process in processes:
                process.join()
            for i in range(self.pool_size):
                next_gen.append(results[i])
            sorted_next_gen = copy.copy(next_gen)
            sorted_next_gen.sort(key=lambda c: c.fitness, reverse=False)
            for child in sorted_next_gen:
                if child.fitness > best_parent.fitness:
                    child = self.__local_optimization(child)
                    yield False, child
                    best_parent = child
                    best_parent.lineage = []
                    historical_fitnesses.append(child.fitness)
                    last_improvement_gen = gen
                    last_improvement_time = time.time()
            for pindex in range(self.pool_size):
                if next_gen[pindex].fitness < parents[pindex].fitness:
                    if self.max_age is None:
                        continue
                    parents[pindex].age += 1
                    if parents[pindex].age < self.max_age:
                        continue
                    index = bisect_left(historical_fitnesses, next_gen[pindex].fitness, 0, len(historical_fitnesses))
                    difference = len(historical_fitnesses) - index
                    proportionSimilar = difference / len(historical_fitnesses)
                    if random.random() < exp(-proportionSimilar):
                        next_gen[pindex].age = parents[pindex].age
                        parents[pindex] = next_gen[pindex]
                        continue
                    parents[pindex] = copy.deepcopy(best_parent)
                    parents[pindex].age = 0
                    continue
                if not next_gen[pindex].fitness > parents[pindex].fitness:
                    next_gen[pindex].age = parents[pindex].age + 1
                    parents[pindex] = next_gen[pindex]
                    continue
                parents[pindex] = next_gen[pindex]
                parents.sort(key=lambda p: p.fitness, reverse=True)
