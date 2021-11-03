import time
from bisect import bisect_left
from math import exp
import multiprocessing as mp
from typing import Any, Tuple, TypeAlias
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
import random
import copy
import os


"""Genetic algorithm engine
"""


numeric = int | float
Genes = TypeAlias
Fitness = TypeAlias


class Chromosome:
    """Object that represents the candidates
    """
    Chromosome = TypeAlias
    __slots__ = ('genes', 'fitness', 'strategy', 'age', 'lineage', 'label')
    
    def __init__(self, genes: Any = None, fitness: Any = None, strategy: list[Callable[[Chromosome], Genes]] = [], 
        age: int = 0, lineage: list = [], label: str = None) -> None:
        """Initializes the Chromosome object

        :param genes: What is wanted to optimize
        :type genes: Any, optional
        :param fitness: Value which describes how much the genes fits what is wanted. It can be of any type since it
            can be compared with > and < and can be printed
        :type fitness: Any, optional
        :param strategy: Container which stores the functions used to obtain the current Chromosome, defaults to []
        :type strategy: list[Callable[[Chromosome], Genes]], optional
        :param age: How many times the candidate was modificated without having any improvement, defaults to 0
        :type age: int, optional
        :param lineage: The historic of Chromosomes used to find the current Chromosome, defaults to []
        :type lineage: list[Chromosome], optional
        :param label: A tag which can be used to identify the Chromosome object, defaults to None, defaults to None
        :type label: str, optional
        """        
        self.genes = genes
        self.fitness = fitness
        self.strategy = strategy
        self.age = age
        self.lineage = lineage
        self.label = label

    @property
    def strategy_str(self) -> str:
        """Returns a string which represents the list of the functions used to obtain the current Chromosome object

        :return: String of a list of the names of the functions used to obtain the current Chromosome object
        :rtype: str
        """              
        return str([strategy.__name__ for strategy in self.strategy]).replace("'", "")


class Mutate:
    """A container for storing mutation functions and its rates. When called it receives a Chromosome and returns
    a new Chromosome

    This is supposed to receive functions which receives a Chromosome object and returns its genes with some random 
    modification

    :return: Mutated Chromosome
    :rtype: Chromosome
    """
    __slots__ = ('methods', 'rate')
    
    def __init__(self, methods: list[Callable[[Chromosome], Genes]], methods_rate: list[numeric]) -> None:
        """Initializes the Mutate object by receiving its parameters

        :param methods: Functions which receives a Chromosome object and returns a new genes
        :type methods: list[Callable[[Chromosome], Genes]]
        :param methods_rate: The rate the functions tends be randomly chosen when the Mutate object is called. It must 
            have the same lenght as methods. Suppose methods is [m1, m2, m3] and methods_rate is  [1, 2, 3]. m2 tends to
            be chosen twice the m1 is and m3 thrice the m1 is
        :type methods_rate: list[numeric]
        """
        self.methods = methods
        self.rate = methods_rate

    def __call__(self, parent: Chromosome, donor: Any = None, mutate_after_crossover: Any = None, 
        mutate_methods: Any = None, first_parent: Any = None) -> Chromosome:
        """Makes the Mutate object callable receiving a Chromosome and returning a Chromosome

        It only deals with the parameter parent. All the remaining parameters can receive absolutely anything without 
        changing the result. It randomly choices one of the methods and returns the result of passing parent  to it. The
        random choice takes into consideration the methods_rate of the current object

        :param parent: Parent Chromosome which will be passed to mutation functions
        :type parent: Chromosome
        :rtype: Chromosome
        """
        child = Chromosome()
        method = random.choices(self.methods, self.rate)[0]
        child.genes = method(parent)
        child.strategy.append(method)
        return child


class Crossover:
    """A container for storing crossover functions and its rates. When called it receives two Chromosome objects and 
    returns a new Chromosome

    This class is supposed to receive functions which receive two Chromosome objects and returns a random combination of
    their genes

    :return: Child Chromosome
    :rtype: Chromosome
    """
    __slots__ = ('methods', 'rate')

    def __init__(self, methods: list[Callable[[Chromosome], Genes]], methods_rate: list[numeric]) -> None:
        """Initializes the Crossover object by receiving its parameters

        :param methods: Functions which receives two Chromosome objects and returns a new genes
        :type methods: list[Callable[[Chromosome]
        :param methods_rate: The rate the functions tends be randomly chosen when the Crossover object is called. It 
            must have the same lenght as methods. Suppose methods is [m1, m2, m3] and methods_rate is  [1, 2, 3]. m2 
            tends to be chosen twice the m1 is and m3 thrice the m1 is.
        :type methods_rate: list[numeric]
        """
        self.methods = methods
        self.rate = methods_rate

    def __call__(self, parent: Chromosome, donor: Chromosome, mutate_after_crossover: bool, mutate_methods: Mutate, 
        first_parent: Chromosome = None) -> Chromosome:
        """Makes the Crossover object callable by receiving its input parameters and returning a new Chromosome

        The parameter first_parent doesn't affect this method. It can receive absolutelly anything without changing the
        result. It randomly choices one of the methods and returns the result of passing parent and donor to it. The 
        random choice takes into consideration the methods_rate of the current object

        :param parent: Parent Chromosome
        :type parent: Chromosome
        :param donor: Chromosome which genes are supposed to be combined with parent's genes
        :type donor: Chromosome
        :param mutate_after_crossover: If its True, after Crossover the child Chromosome will be passed to 
            mutate_methods before return it, if its False so it will be returned immediatly after crossover
        :type mutate_after_crossover: bool
        :param mutate_methods: Mutate object which child Chromosome will be passade if mutate_after_crossover is True
        :type mutate_methods: Mutate
        :return: Child Chromosome which genes are supposed to be a combination between parent's and donor's
        :rtype: Chromosome
        """
        child = Chromosome()
        method = random.choices(self.methods, self.rate)[0]
        child.lineage += donor.lineage
        child.lineage.append(donor)
        child.genes = method(parent, donor)
        child.strategy.append(method)
        if mutate_after_crossover:
            return mutate_methods(child)
        return child


class Create:
    """A container for storing genes creation functions and its rates. When called it receives the first_parent 
    Chromosome and returns a new Chromosome

    This class is supposed to receive functions which receive a generic parent Chromosome and return a whole new genes 
    without any bound to the parent's one. The exceptions are mutate_best and mutate_first, which respectively returns
    a result of passing the best and the first parent, respectively to the mutate object

    :return: New Chromosome object with the created genes
    :rtype: Chromosome
    """
    __slots__ = ('methods', 'rate')

    def __init__(self, methods: list[Callable[[Chromosome], Genes]], methods_rate: list[numeric]) -> None:
        """Initializes the Crossover object by receiving its parameters

        :param methods: Functions which receives the first_parent Chromosome and returns a new genes
        :type methods: list[Callable[[Chromosome], Genes]]
        :param methods_rate: The rate the functions tends be randomly chosen when the Crossover object is called. It 
            must have the same lenght as methods. Suppose methods is [m1, m2, m3] and methods_rate is  [1, 2, 3]. m2 
            tends to be chosen twice the m1 is and m3 thrice the m1 is.
        :type methods_rate: list[numeric]
        """
        self.methods = methods
        self.rate = methods_rate

    def __call__(self, parent: Chromosome = None, donor: Chromosome = None, mutate_after_crossover: bool = None, 
        mutate_methods: Mutate = None, first_parent: Chromosome = None) -> Chromosome:
        """Makes the Create method callable by receiving the first_parent input parameter and returning a new Chromosome

        It only deals with the parameter first_parent. All the remaining parameters can receive absolutely anything 
        without changing the result. It randomly choices one of the methods and returns the result of passing  
        first_parent to it. The random choice takes into consideration the methods_rate of the current object

        :param first_parent: It's designed to receive the first candidate created
        :type first_parent: Chromosome
        :return: New Chromosome object with the created genes
        :rtype: Chromosome
        """
        child = Chromosome()
        method = random.choices(self.methods, self.rate)[0]
        child.genes = method(first_parent)
        child.strategy.append(method)
        return child


class Strategies:
    """A container for sotoring the candidate generation strategies (Create, Mutate and Crossover objects) and its rates

    It must receive only one of each generation strategy (Create, Mutate and Crossover) object.

    :return: New Chromosome object with the genes generated by the strategy randomly selected
    :rtype: Chromosome
    """    
    __slots__ = ('strategies', 'rate')

    def __init__(self, strategies: Tuple[Mutate, Crossover, Create], strategies_rate: list[numeric]) -> None:
        """Initializes the Strategies object by receiving its parameters

        :param strategies: A list with Create, Mutate and Crossover objects
        :type strategies: Tuple[Create, Mutate, Crossover]
        :param strategies_rate: The rate the Mutate, Crossover and Create objects tends be randomly chosen when the 
            Strategies object is called. It must have the same lenght as methods. Suppose strategies is [s1, s2, s3] and 
            strategies_rate is  [1, 2, 3]. s2 tends to be chosen twice the s1 is and s3 thrice the s1 is.
        :type strategies_rate: list[numeric]
        """
        self.strategies = strategies
        self.rate = strategies_rate

    def __call__(self, parent: Chromosome, donor: Chromosome, mutate_after_crossover: bool, mutate_methods: Mutate, 
        first_parent: Chromosome) -> Chromosome:
        """Makes the Create method callable by receiving its input parameters and returning a new Chromosome

        Randomly choices one of the methods and returns the result of passing the inputs to it. The random choice takes
        into consideration the strategies_rate of the current object.

        :param parent: Parent Chromosome
        :type parent: Chromosome
        :param donor: Chromosome which genes are supposed to be combined with parent's genes
        :type donor: Chromosome
        :param mutate_after_crossover: If its True, after Crossover the child Chromosome will be passed to 
            mutate_methods before return it, if its False so it will be returned immediatly after crossover
        :type mutate_after_crossover: bool
        :param mutate_methods: Mutate object which child Chromosome will be passade if mutate_after_crossover is True
        :type mutate_methods: Mutate
        :param first_parent: It's designed to receive the first candidate created
        :type first_parent: Chromosome
        :return: New Chromosome object with the genes generated by the strategy randomly selected
        :rtype: Chromosome
        """        
        return random.choices(self.strategies, self.rate)(parent, donor, mutate_after_crossover, mutate_methods,
            first_parent)


def mutate_first(first_parent: Chromosome) -> Genes:
    """Creation function that receives the Chromosome of the first created parent and pass it to the Mutate object to 
    return the result

    This function is actually a void, it exists only to be called by import and to be override by the actual 
    mutate_first function

    :param first_parent: First created candidate
    :type first_parent: Chromosome
    :return: Mutated genes of the first candidate
    :rtype: genes
    """
    pass


def mutate_best(best_candidate: Chromosome) -> Genes:
    """Creation function that receives the Chromosome of the best candidate and pass it to the Mutate object to return 
    the result

    This function is actually a void, it exists only to be called by import and to be override by the actual mutate_best
    function

    :param best_candidate: Best candidate
    :type best_candidate: Chromosome
    :return: Mutated genes of the best candidate
    :rtype: genes
    """
    pass


class Genetic(ABC):
    """Genetic algorithm abstract class

    This abstract class provides a framework for creating problem-specific genetic algorithms. To use it you must create
    a class that inherits it. The class that inherits it must have at least two methods: get_fitness and save.
    
    :param ABC: Helper class that provides a standard way to create an abstract class using inheritance.
    :type ABC: class
    """
    __slots__ = ('first_genes', 'fitness_param', 'strategies', 'max_age', 'pool_size', 'mutate_after_crossover', 
        'crossover_elitism', 'elitism_rate', 'freedom_rate', 'parallelism', 'local_opt', 'max_seconds', 'time_toler', 
        'gens_toler', 'max_gens', 'save_directory', 'start_time', 'first_parent', 'lineage_ids', 'best_candidate', 
        'mutate_methods', 'create_methods', 'crossover_methods')

    def __init__(self, first_genes: Chromosome, strategies: Strategies, max_age: int | None, pool_size: int, 
        mutate_after_crossover: bool, crossover_elitism: list[numeric] | None, elitism_rate: list[int] | None, 
        freedom_rate: int, parallelism: bool, local_opt: bool, max_seconds: numeric | None, time_toler: numeric | None, 
        gens_toler: numeric | None, max_gens: numeric | None, save_directory: str) -> None:
        """Initializes the Genetic object by receiving its parameters

        :param first_genes: The genes of the first candidate in the genetic algorithm
        :type first_genes: Chromosome
        :param strategies: Strategies object
        :type strategies: Strategies
        :param max_age: The max amount of times a Chromosome can suffer chaging strategies without improve its fitness
        :type max_age: int
        :param pool_size: The amount of candidates being optimized simultaneously
        :type pool_size: int
        :param mutate_after_crossover: If it's True, than after each crossover operation, the resultant child Chromosome
            suffer a mutate operation before return to the genetic algorithm, but if it's false the mutation doesn't
            occur and the child Chromosome is returned immediately after the crossover
        :type mutate_after_crossover: bool
        :param crossover_elitism: The rate each candidate tends to be selected to be the gene's donor in any crossover 
            operation from the best to the worst. Its lenght must be equal pool_size value. If pool_size is 3 and 
            crossover_elitism is [3, 2, 1] the best candidate has the triple of the chance to be selected than the 
            worst, the medium candidate has double. It can also receive None, and it means all candidates are equally
            probable to be selected for being the genes' donor on a crossover
        :type crossover_elitism: list[numeric] | None
        :param elitism_rate: List of reprodution rate of each candidate, from the best to the worst. the sum of its 
            elements also must be less or equal than pool_size. If pool_size is 16 and elitism_rate is [4, 3, 2] it 
            means the best candidate in the current generation's pool of candidates will provide 4 descendants for the 
            next generation, the second best will provide 3 and the third best will provide two, then then remain 7 
            available spaces in the next generation's pool will be filled with one descendant of each of the next seven 
            candidates in this order
        :type elitism_rate: list[int] | None
        :param freedom_rate: The number of candidate generation strategies (Mutate, Crossover and Create) the candidate 
            will suffer aways a new candidate is needed to be generated (if Create is selected it means the candidate is
            supposed to be substituted by a whole new one without any relation with the parent candidate)
        :type freedom_rate: int
        :param parallelism: If it's True than each fitness calculation will be done in a different process, what changes
            the whole dynamics of the genetic algorithm. With paraellism enabled, the concept of generations emerges as
            we can have different candidates being caculated at the same time. If it's False, there will be no
            generations and candidates' fitnesses will be calculated sequentially
        :type parallelism: bool
        :param local_opt: If its True makes that every time the algorithm genetic gets a new best candidate it is sent 
            to the local_optimize function (which in this case must be override by the subclass) that is supposed to
            perform a local optimization (in the solution-space of the specific problem) over the genes of the best 
            candidate, and then return a new best candidate with the optimized genes
        :type local_opt: bool
        :param max_seconds: The max amount of seconds the genetic algorithm can run. Once exceeded this amount, the
            the running will be stoped and the best candidate will be returned. It can also receive None, and in this 
            case this limit wouldn't exist
        :type max_seconds: numeric | None
        :param time_toler: The max amount of seconds the algorithm can still running without has any improvements on its
            best candidate's fitness. Once exceeded this amount, the running will be stoped and the best candidate will 
            be returned. It can also receive None, and in this case this limit wouldn't exist
        :type time_toler: numeric | None
        :param gens_toler: The maximum amount of generations the algorithm genetic can run in sequence without having
            any improvement on it's best parent fitness. It can also receive None and in this case this limit wouldn't 
            exist. It only works when parallelism is True, otherwise it doesn't affect anything
        :type gens_toler: numeric | None
        :param max_gens: The max amount of generations the genetic algorithm can run. Once exceeded this amount, the
            the running will be stoped and the best candidate will be returned. It can also receive None, and in this 
            case this limit wouldn't exist. It only works when parallelism is True, otherwise it doesn't affect anything
        :type max_gens: numeric | None
        :param save_directory: The directory address relative to __main__ where the outputs will be saved. If its None
            than it will receive the instant of time the running started
        :type save_directory: str
        """        
        self.first_genes = first_genes
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
        self.save_directory = save_directory
        self.start_time = None
        self.first_parent = None
        self.lineage_ids = []
        self.best_candidate = None
        for strategy in strategies.strategies:
            if type(strategy) is Mutate:
                self.mutate_methods = strategy
            elif type(strategy) is Create:
                self.create_methods = strategy
            elif type(strategy) is Crossover:
                self.crossover_methods = strategy
        if mutate_first in self.create_methods.methods:
            self.create_methods.methods[self.create_methods.methods.index(mutate_first)] = self.mutate_first
        if mutate_best in self.create_methods.methods:
            self.create_methods.methods[self.create_methods.methods.index(mutate_best)] = self.mutate_best
        
    @abstractmethod
    def get_fitness(self, candidate: Chromosome) -> Fitness:
        """Abstract method which must be override by one which receives a candidate's Chromosome and returns it's
            fitness

        :param candidate: Candidate wich fitness is wanted
        :type candidate: Fitness
        :type parent: Candidate's fitness
        :rtype: Fitness
        """
        pass

    @abstractmethod
    def save(self, candidate: Chromosome, file_name: str, directory: str) -> None:
        """Abstract method which must be override by one which receives a candidate, a file_name and a directory

        This method which overrides it must receive a candidate's Chromosome, a string which is the name of the file
        the candidate will be saved and a string which is the directory where it will be saved than this function must
        save the candidate relevant informations in a document named as given in the directory given. The directory is
        given relative to __main__

        :param candidate: Candidate which is wanted to save
        :type candidate: Chromosome
        :param file_name: Name of the file in which the candidate's informations must be saved
        :type file_name: str
        :param directory: Directory where the file with the candidate's information must be saved
        :type directory: str
        """
        pass

    def mutate_first(self, first_parent: Chromosome) -> Genes:
        """The actual mutate_first function

        Creation function that receives the Chromosome of the first candidate and pass it to the Mutate object to return 
        the result

        :param first_parent: First created candidate
        :type first_parent: Chromosome
        :return: Mutated genes of the first candidate
        :rtype: genes
        """
        return self.mutate_methods(self.first_parent).genes

    def mutate_best(self, best_candidate: Chromosome) -> Genes:
        """The actual mutate_best function

        Creation function that receives the Chromosome of the best candidate and pass it to the Mutate object to return 
        the result

        :param best_candidate: Best candidate
        :type best_candidate: Chromosome
        :return: Mutated genes of the best candidate
        :rtype: genes
        """
        return self.mutate_methods(self.best_candidate).genes

    @staticmethod
    def catch(candidate: Chromosome) -> None:
        """Static method which will be executed if an error occurs during a candidate generation

        It can be override if needed. By default it just raises an exception

        :param candidate: Candidate which was being generated while error occurs
        :type candidate: Chromosome
        :raises Exception: Raises exception if error occurs during candidate generation
        """
        raise Exception(f'An exception ocurred while generating candidate {candidate.label}.')

    @staticmethod
    def display(candidate: Chromosome, timediff: float) -> None:
        """Generate what is printed on console everytime a new best candidate is reached

        It can be override if needed

        :param candidate: Best candidate founded
        :type candidate: Chromosome
        :param timediff: Time difference between the candidate is founded and the start of the execution
        :type timediff: float
        """
        print("{0}\t{1}".format((candidate.fitness), str(timediff)))

    def load(self) -> Chromosome:
        """Returns a candidate with the first_genes and the fitness of the first_parent

        :return: First candidate
        :rtype: Chromosome
        """
        return Chromosome(self.first_genes, self.first_parent.fitness, [self.load])

    def __get_child(self, candidates: list[Chromosome], parent_index: int, queue: mp.Queue = None, child_index: int = None, 
        label: str = None):
        """Returns a new candidate

        Tries to generate a new candidate inside a 'while True' loop with a 'try except' statement which is broke when
        the new candidate is successfully generated. If an exception occurs during the candidate generation try, the 
        function 'catch' will be called with such candidate as argument

        :param candidates: Pool of candidates
        :type candidates: list[Chromosome]
        :param parent_index: Index of the current parent's Chromosome in candidates
        :type parent_index: int
        :param queue: Multiprocessing queue by which the candidates are returned in case of parallelism, defaults to 
            None
        :type queue: mp.Queue, optional
        :param child_index: Index of child in the next generation pool. It's used to identify the returned chromosome in
            the main process in case of parallelism, defaults to None
        :type child_index: int, optional
        :param label: Child's Chromosome's label, defaults to None
        :type label: str, optional
        :return: Child
        :rtype: Chromosome
        """
        sorted_candidates = copy.copy(candidates)
        sorted_candidates.sort(reverse=True, key=lambda p: p.fitness)
        while True:
            try:
                parent = copy.deepcopy(candidates[parent_index])
                for _ in range(self.freedom_rate):
                    donor = random.choices(sorted_candidates, self.crossover_elitism)[0]
                    child = self.strategies(parent, donor, self.mutate_after_crossover, self.mutate_methods, 
                        self.first_parent)
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

    def __local_optimization(self, candidate: Chromosome) -> Chromosome:
        """Receives a candidate then performs a local optimization on its genes and so returns a new Chromosome with
        these new genes and its fitness

        :param candidate: Candidate which genes are wanted to be optimized
        :type candidate: Chromosome
        :return: Optimized candidate
        :rtype: Chromosome
        """
        if not self.local_opt:
            return candidate
        new_genes = self.local_optimize(candidate)
        new_fitness = self.get_fitness(candidate)
        return Chromosome(new_genes, new_fitness, candidate.strategy + [self.local_optimize], 0, 
            candidate.lineage + [candidate])

    def local_optimize(self, candidate: Chromosome) -> Genes:
        """This method must be override in case of local_opt is True for a method which receives a candidate's 
        Chromosome and returns its genes locally optimized

        :param candidate: Candidate which genes are wanted to be optimized
        :type candidate: Chromosome
        :raises Exception: Raises exception if this method was not override by the subclass
        :return: Optimized genes
        :rtype: Genes
        """        
        raise Exception(f'The class {self.__class__.__name__} must override the local_optimize method of parent class')

    def __generate_parent(self, queue: mp.Queue = None, label: str = None):
        """Uses the Create object from strategies to generate a new candidate

        Tries to generate a new candidate using the Create object from strategies inside a 'while True' loop with a 'try
        except' statement which is broke when the new candidate is successfully generated. If an exception occurs during
        the candidate generation try, the function 'catch' will be called with such candidate as argument

        :param queue: Multiprocessing queue by which the candidates are returned in case of parallelism, defaults to 
            None
        :type queue: mp.Queue, optional
        :param label: New parent's Chromosome's label, defaults to None
        :type label: str, optional
        :return: New candidate
        :rtype: Chromosome
        """
        while True:
            try:
                parent = self.create_methods(first_parent=copy.deepcopy(self.first_parent))
                parent.label = label
                parent.fitness = self.get_fitness(parent)
                break
            except:
                self.catch(parent)
        if queue is not None:
            queue.put(parent)
        return parent

    def run(self) -> Chromosome:
        """Starts the genetic algorithm execution

        :return: Best candidate
        :rtype: Chromosome
        """
        self.save_directory = f"{datetime.now().strftime('%Y_%m_%d_%H:%M')}" if self.save_directory is None \
            else self.save_directory
        os.mkdir(self.save_directory)
        os.mkdir(f'{self.save_directory}/lineage')
        os.mkdir(f'{self.save_directory}/improvements')
        self.start_time = time.time()
        self.first_parent = Chromosome(self.first_genes, label = '0_0')
        self.first_parent.fitness = self.get_fitness(self.first_parent)
        opt_func = self.__get_improvement_mp if self.parallelism else self.__get_improvement
        n = 0
        j = 0
        for timed_out, improvement in opt_func():
            self.save(improvement, f'{n}_{improvement.label}', f'{self.save_directory}/improvements')
            improvement.lineage.append(improvement)
            self.best_candidate = improvement
            timediff = time.time() - self.start_time
            self.display(improvement, timediff)
            with open(f'{self.save_directory}/improvements_strategies.log', 'a') as islog:
                islog.write(f'{improvement.strategy_str}\t{improvement.fitness}\t{timediff}\n')
            with open(f'{self.save_directory}/lineage_strategies.log', 'a') as lslog:
                for ancestor in improvement.lineage:
                    if not ancestor.label in self.lineage_ids:
                        self.lineage_ids.append(ancestor.label)
                        lslog.write(f'{ancestor.strategy_str}\t{ancestor.fitness}\t{timediff}\n')
                        self.save(ancestor, f'{j}_{ancestor.label}', f'{self.save_directory}/lineage')
                        j += 1
            n += 1
            if timed_out:
                break
        self.save(improvement, f'{n}_{improvement.label}_best', f'{self.save_directory}')
        with open(f'{self.save_directory}/strategies_log.txt', 'a') as slog:
            slog.write('\n---')
        return improvement

    def __get_improvement(self):
        """Generator of genetic improvements

        :yield: Best candidate achieved until the moment
        :rtype: Chromosome
        """
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
        """Genetic improvements generator using multiprocessing

        :raises Exception: Raises an exception if elitism rate is incompatible with pool size
        :yield: Best candidate achieved until the moment
        :rtype: Chromosome
        """
        elit_size = len(self.elitism_rate) if self.elitism_rate is not None else None
        last_improvement_time = self.start_time
        queue = mp.Queue(maxsize=self.pool_size - 1)
        processes = []
        gen = 0
        if elit_size is not None:
            if sum(self.elitism_rate) > self.pool_size:
                raise Exception('Elitism exceeds pool size.')
        best_parent = self.__local_optimization(self.load())
        best_parent.label = '0_0'
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
