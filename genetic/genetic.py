from __future__ import annotations
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
import inspect
import json


"""Genetic algorithm engine
"""


Genes: TypeAlias = 'Genes'
Fitness: TypeAlias = 'Fitness'


class Chromosome:
    """Class that represents the candidates
    """
    __slots__ = ('genes', 'fitness', 'strategy', 'age', 'lineage', 'label')
    
    def __init__(self, genes: Genes = None, fitness: Fitness = None, strategy: list[Callable[[Chromosome], Genes]] = [], 
        age: int = 0, lineage: list = [], label: str = None) -> None:
        """Initializes the Chromosome object

        Parameters
        ----------
        genes : Genes, optional
            What is wanted to optimize, by default None
        fitness : Fitness, optional
            Value which describes how much the genes fits what is wanted. It can be of any type since it can be compared
            with > and < and can be printed, by default None
        strategy : list[Callable[[Chromosome], Genes]], optional
            Container which stores the functions used to obtain the current Chromosome, by default []
        age : int, optional
            How many times the candidate was modificated without having any improvement, by default 0
        lineage : list, optional
            The historic of Chromosomes used to find the current Chromosome, by default []
        label : str, optional
            A tag which can be used to identify the Chromosome object, by default None
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

        Returns
        -------
        str
            String of a list of the names of the functions used to obtain the current Chromosome object
        """
        return str([strategy.__name__ for strategy in self.strategy]).replace("'", "")

    def copy(self):
        return copy.deepcopy(self)


class Mutate:
    """A container for storing mutation functions and its rates
    """
    __slots__ = ('methods', 'rate')
    
    def __init__(self, methods: list[Callable[[Chromosome, dict], Genes]], methods_rate: list[float]) -> None:
        """Initializes the Mutate object

        Parameters
        ----------
        methods : list[Callable[[Chromosome, dict], Genes]]
            Functions which receives a Chromosome object and returns a new genes
        methods_rate : list[float]
            The rates the functions tends be randomly chosen when the Mutate object is called
        """
        self.methods = methods
        self.rate = methods_rate

    def __call__(self, attributes: dict, parent: Chromosome, donor = None) -> Chromosome:
        """Makes the Mutate object a callable one that receives the attributes dict and a Chromosome and returns a new
        Chromosome

        Parameters
        ----------
        attributes : dict
            Global parameters
        parent : Chromosome
            Parent chromosome

        Returns
        -------
        Chromosome
            Mutated chromosome
        """
        method = random.choices(self.methods, self.rate)[0]
        child = Chromosome(genes=method(parent.genes, attributes), fitness=None, strategy=parent.strategy + [method],
            age=0, lineage=[])
        return child


class Crossover:
    """A container for storing crossover functions and its rates
    """
    __slots__ = ('methods', 'rate')

    def __init__(self, methods: list[Callable[[Chromosome], Genes]], methods_rate: list[float]) -> None:
        """Initializes the Crossover object

        Parameters
        ----------
        methods : list[Callable[[Chromosome], Genes]]
            Functions which receives the attributes dict, two Chromosome objects and returns a new genes
        methods_rate : list[float]
            The rates the functions tends be randomly chosen when the Crossover object is called
        """
        self.methods = methods
        self.rate = methods_rate

    def __call__(self, attributes: dict, parent: Chromosome, donor: Chromosome) -> Chromosome:
        """Makes the Crossover object a callable one that receives its parameters and returns a new Chromosome

        Parameters
        ----------
        attributes : dict
            Global parameters
        parent : Chromosome
            Parent chromosome
        donor : Chromosome
            Donor chromosome

        Returns
        -------
        Chromosome
            Child chromosome
        """
        method = random.choices(self.methods, self.rate)[0]
        child = Chromosome(genes=method(parent.genes, donor.genes, attributes), fitness=None,
            strategy=parent.strategy + [method], age=0, lineage=[])
        return child


class Create:
    """A container for storing genes creation functions and its rates
    """
    __slots__ = ('methods', 'rate')

    def __init__(self, methods: list[Callable[[Chromosome], Genes]], methods_rate: list[float]) -> None:
        """Initializes the Crossover object

        Parameters
        ----------
        methods : list[Callable[[Chromosome], Genes]]
            Functions which receives the attributes dict and returns a new genes
        methods_rate : list[float]
            The rates the functions tends be randomly chosen when the Crossover object is called
        """
        self.methods = methods
        self.rate = methods_rate

    def __call__(self, attributes: dict, parent = None, donor = None) -> Chromosome:
        """Makes the Create method a callable one that receives the attributes dict and returning a new chromosome

        Parameters
        ----------
        attributes : dict
            Global parameters

        Returns
        -------
        Chromosome
            New chromosome
        """
        method = random.choices(self.methods, self.rate)[0]
        child = Chromosome(genes=method(attributes), fitness=None, strategy=[method], age=0, lineage=[])
        return child


class Strategies:
    """A container for storing the candidate generation strategies (Create, Mutate and Crossover objects) and its rates
    """
    __slots__ = ('strategies', 'rate')

    def __init__(self, strategies: Tuple[Mutate, Crossover, Create], strategies_rate: list[float]) -> None:
        """Initializes the Strategies object

        Parameters
        ----------
        strategies : Tuple[Mutate, Crossover, Create]
            A list with Create, Mutate and Crossover objects
        strategies_rate : list[float]
            The rates the Mutate, Crossover and Create objects tends be randomly chosen when the Strategies object is
            called
        """
        self.strategies = strategies
        self.rate = strategies_rate


class Genetic(ABC):
    """Genetic algorithm abstract class
    """
    __slots__ = ('first_genes', 'strategies', 'max_age', 'pool_size', 'mutate_after_crossover', 'crossover_elitism',
        'elitism_rate', 'freedom_rate', 'opt_func', 'max_seconds', 'time_toler', 'gens_toler', 'max_gens',
        'max_fitness', 'save_directory', 'start_time', 'first_parent', 'lineage_ids', 'best_candidate',
        'aways_local_opt', 'time_toler_opt', 'gens_toler_opt', 'gen', 'historical_fitnesses', 'attributes',
        'create_methods', 'mutate_methods', 'crossover_methods', 'last_improvement_time', 'improvements_strategies',
        'lineage_strategies', 'improvements_gens_diff', 'adapting_rate', 'improvements_increase', 'adapting_args',
        'last_improvement_gen')

    def __init__(self, first_genes: Genes, strategies: Strategies, max_age: int | None, pool_size: int, 
        mutate_after_crossover: bool, crossover_elitism: list[float] | None,  elitism_rate: list[int] | None,
        freedom_rate: int, parallelism: bool, aways_local_opt: bool, time_toler_opt: float | None,
        gens_toler_opt: int | None, max_seconds: float | None, time_toler: float | None, gens_toler: int | None,
        max_gens: int | None, max_fitness: Fitness | None, save_directory: str | None, adapting_rate: float | None,
        adapting_args: Tuple[str, str] | None, attributes: dict | None) -> None:
        """Initializes the Genetic object

        Parameters
        ----------
        first_genes : Genes
            The genes of the first candidate in the genetic algorithm
        strategies : Strategies
            Strategies object
        max_age : int | None
            The max amount of times a Chromosome can suffer chaging strategies without improve its fitness
        pool_size : int
            The amount of candidates being optimized together
        mutate_after_crossover : bool
            If its True, then after after each crossover operation, the resultant child Chromosome will suffer a mutate
            operation
        crossover_elitism : list[float] | None
            The rates each candidate tends to be selected to be the gene's donor in any crossover operation from the
            best to the worst
        elitism_rate : list[int] | None
            Reprodution rate of each candidate, from the best to the worst
        freedom_rate : int
            The number of candidate generation strategies (Mutate, Crossover and Create) the candidate will suffer aways
            a new candidate is needed to be generated
        parallelism : bool
            If it's True then each fitness calculation will be done in a different process
        aways_local_opt : bool
            If it's True then every time the algorithm genetic gets a new best candidate it is sent to the
            local_optimize function
        time_toler_opt : float | None
            If the time between two improvements is bigger than this, then the new improvement will be sent to the
            local_optimization function
        gens_toler_opt : int | None
            If the generations that passed between two improvements are more than this, then the new improvement will be
            sent to the local_optimization function
        max_seconds : float | None
            The maximum amount of seconds the genetic algorithm can run
        time_toler : float | None
            The maximum amount of seconds the algorithm can still running without having any improvements on its best
            candidate's fitness
        gens_toler : int | None
            The maximum amount of generations the algorithm genetic can run in sequence without having any improvement
            on it's best candidate's fitness
        max_gens : int | None
            The maximum amount of generations the genetic algorithm can run
        max_fitness : Fitness | None
            The maximum fitness the genetic algorithm can achieve
        save_directory : str | None
            The directory address relative to __main__ where the outputs will be saved
        adapting_rate : float | None
            The rate the strategies rates its methods rates will be adapted to the historical rate
        adapting_args : Tuple[str, str] | None
            Arguments that define the adapting function. It must have two strings, one to define the adapting chance and
            one to define the adapting target. The adapting chance possible arguments are "frequency" which increases
            the chance to adapt the lower is the frequency the improvements are achieved and "fitness_difference" which
            increases the chance to adapt the less significance the improvements have. The adapting target possible
            arguments are "fitness_increasement" which privilegies the strategies that most contributed with the fitness
            increasement, "improvements" that privilegies the strategies that generated most best candidates and
            "lineage" that privilegies the strategies that generated more ancestors
        attributes : dict | None
            Global parameters which will be passed to each mutation, crossover and creation operators to be used if
            needed
        """
        self.aways_local_opt = aways_local_opt
        self.gens_toler_opt = gens_toler_opt
        self.time_toler_opt = time_toler_opt
        self.crossover_elitism = crossover_elitism
        self.save_directory = save_directory
        self.strategies = copy.deepcopy(strategies)
        self.max_age = max_age
        self.pool_size = pool_size
        self.mutate_after_crossover = mutate_after_crossover
        self.elitism_rate = elitism_rate
        self.freedom_rate = freedom_rate
        self.max_seconds = max_seconds
        self.time_toler = time_toler
        self.gens_toler = gens_toler
        self.max_gens = max_gens
        self.max_fitness = max_fitness
        self.attributes = attributes
        self.save_directory = save_directory
        self.adapting_rate = adapting_rate
        self.adapting_args = adapting_args
        self.attributes.update({'first_candidate': Chromosome(first_genes, None, [self.load], 0, [], '0_0')})
        self.attributes.update({'best_candidate': self.attributes['first_candidate']})
        self.start_time = None
        self.lineage_ids = []
        for strategy in self.strategies.strategies:
            if type(strategy) is Mutate:
                self.mutate_methods = strategy
            elif type(strategy) is Create:
                self.create_methods = strategy
            elif type(strategy) is Crossover:
                self.crossover_methods = strategy
        if Genetic.mutate_first in self.create_methods.methods:
            self.create_methods.methods[self.create_methods.methods.index(Genetic.mutate_first)] = self.mutate_first
        if Genetic.mutate_best in self.create_methods.methods:
            self.create_methods.methods[self.create_methods.methods.index(Genetic.mutate_best)] = self.mutate_best
        self.last_improvement_time = None
        self.last_improvement_gen = None
        self.gen = None
        self.historical_fitnesses = []
        if type(parallelism) == bool:
            self.opt_func = self.__get_improvement_mp if parallelism else self.__get_improvement
        else:
            self.opt_func = None
        self.improvements_strategies = {strategy.__class__.__name__:\
            {method.__name__: 0 for method in strategy.methods} for strategy in self.strategies.strategies}
        self.lineage_strategies = {strategy.__class__.__name__:\
            {method.__name__: 0 for method in strategy.methods} for strategy in self.strategies.strategies}
        self.improvements_increase = {strategy.__class__.__name__:\
            {method.__name__: 0 for method in strategy.methods} for strategy in self.strategies.strategies}
        self.improvements_gens_diff = []

    def improvements_diff(self) -> list[float]:
        """List of differences between each consecutive best candidate's fitness

        Returns
        -------
        list[float]
            List of differences between each consecutive best candidate's fitness
        """
        return [self.historical_fitnesses[0] - self.attributes['first_candidate'].fitness] +\
            [self.historical_fitnesses[i] - self.historical_fitnesses[i-1] for i in\
            range(1, len(self.historical_fitnesses))]

    @abstractmethod
    def get_fitness(self, candidate: Chromosome) -> Fitness:
        """Abstract method which must be override by one which receives a candidate's Chromosome and returns it's
        fitness

        Parameters
        ----------
        candidate : Chromosome
            Candidate wich fitness is wanted

        Returns
        -------
        Fitness
            Candidate's fitness
        """
        pass

    @abstractmethod
    def save(self, candidate: Chromosome, file_name: str, directory: str) -> None:
        """Abstract method which must be override by one which saves the candidate's data in the file named as given by
        the file_name parameter in the directory given by the directory parameter

        Parameters
        ----------
        candidate : Chromosome
            Candidate which is wanted to save
        file_name : str
            Name of the file in which the candidate's informations must be saved
        directory : str
            Directory where the file with the candidate's data must be saved
        """
        pass

    def mutate_first(self, attributes: dict) -> Genes:
        """Creation function that returns a mutation of the first candidate

        Parameters
        ----------
        attributes : dict
            Global parameters

        Returns
        -------
        Genes
            Mutated genes of the first candidate
        """
        return self.mutate_methods(attributes['first_candidate'])

    def mutate_best(self, attributes: dict) -> Genes:
        """Creation function that returns a mutation of the current best candidate

        Parameters
        ----------
        attributes : dict
            Global parameters

        Returns
        -------
        Genes
            Mutated genes of the current best candidate
        """
        return self.mutate_methods(attributes['best_candidate'])

    def catch(self, candidate: Chromosome) -> None:
        """Method which will be executed if an error occurs during a candidate generation. It can be override if needed

        Parameters
        ----------
        candidate : Chromosome
            Candidate which was being generated while error occurs

        Raises
        ------
        Exception
            An exception ocurred while generating the candidate
        """
        raise Exception(f'An exception ocurred while generating candidate {candidate.label}.')

    @staticmethod
    def display(candidate: Chromosome, timediff: float) -> None:
        """Generate what is printed on console everytime a new best candidate is achieved

        Parameters
        ----------
        candidate : Chromosome
            Best candidate achieved
        timediff : float
            Time difference between the candidate is found and the start of the execution
        """
        print("{0}\t{1}\t{2}".format(candidate.label, timediff,candidate.fitness))

    def load(self) -> Chromosome:
        """Returns a Chromosome with the genes and the fitness of the first_parent

        Returns
        -------
        Chromosome
            Copy of first candidate
        """
        return self.attributes['first_candidate'].copy()

    @staticmethod
    def remove_lineage(candidate: Chromosome) -> Chromosome:
        """Removes the lineage of a candidate

        Parameters
        ----------
        candidate : Chromosome
            Candidate which lineage will be removed

        Returns
        -------
        Chromosome
            Candidate with lineage changed to None
        """
        candidate.lineage = []
        return candidate

    def __get_child(self, candidates: list[Chromosome], parent_index: int, queue: mp.Queue = None,
        child_index: int = None, label: str = None) -> Chromosome:
        """Generate a new candidate

        Parameters
        ----------
        candidates : list[Chromosome]
            Current generation pool
        parent_index : int
            Index of the current parent's Chromosome in candidates
        queue : mp.Queue, optional
            Multiprocessing queue by which the candidates are returned in case of parallelism, by default None
        child_index : int, optional
            Index of child in the next generation pool, by default None
        label : str, optional
            Child's Chromosome's label, by default None

        Returns
        -------
        Chromosome
            Child chromosome
        """
        sorted_candidates = copy.copy(candidates)
        sorted_candidates.sort(reverse=True, key=lambda p: p.fitness)
        donors = list(zip(copy.copy(sorted_candidates), self.crossover_elitism))
        while True:
            try:
                current_donors = copy.copy(donors)
                parent = Chromosome(copy.deepcopy(candidates[parent_index].genes))
                current_donors = list(filter(lambda x: x[0] != parent, current_donors))
                lineage = list(map(self.remove_lineage, parent.lineage))
                for _ in range(self.freedom_rate):
                    strategy = random.choices(self.strategies.strategies, self.strategies.rate)[0]
                    if isinstance(strategy, Crossover):
                        donor = random.choices([donor[0] for donor in current_donors],
                            [donor[1] for donor in current_donors])[0]
                        current_donors = list(filter(lambda x: x[0] != donor, current_donors))
                    else:
                        donor = None
                    child = strategy(self.attributes, parent, donor)
                    if isinstance(strategy, Crossover):
                        lineage += list(map(self.remove_lineage, copy.copy(donor.lineage)))
                        if self.mutate_after_crossover:
                            child = self.mutate_methods(self.attributes, child)
                    parent = child
                lineage += [child]
                child.lineage = lineage
                child.label = label
                child.lineage = list(set(child.lineage))
                child.lineage = [ancestor for ancestor in child.lineage if ancestor.label not in self.lineage_ids]
                child.fitness = self.get_fitness(child)
                break
            except:
                self.catch(child)
        if queue is not None:
            queue.put({child_index: child})
        return child

    def __local_optimization(self, candidate: Chromosome) -> Chromosome:
        """Performs a local optimization on its genes and then returns a new Chromosome with these new genes and its
        fitness

        Parameters
        ----------
        candidate : Chromosome
            Candidate which genes are wanted to be optimized

        Returns
        -------
        Chromosome
            Optimized candidate
        """
        if ((self.time_toler_opt is not None and time.time() - self.last_improvement_time > self.time_toler_opt) or\
            (self.gens_toler_opt is not None and self.gen - self.last_improvement_gen() > self.gens_toler_opt) or\
                self.aways_local_opt):
            opt_candidate = Chromosome(genes=self.local_optimize(candidate), fitness=None,
                strategy=candidate.strategy + [self.local_optimize], age=0, lineage=candidate.lineage,
                label=candidate.label)
            opt_candidate.lineage += [opt_candidate]
            opt_candidate.fitness = self.get_fitness(opt_candidate)
            return opt_candidate
        return candidate

    def update_parameters(self) -> None:
        """Calls update_function sending the genetic object if there was some improvement in the last generation. This
        function can be override if is needed to update the execution parameters at the end of each generation
        """
        if self.adapting_rate is not None and self.adapting_args is not None\
            and self.last_improvement_gen == self.gen:
            self.adapt_rates()

    def adapt_rates(self) -> None:
        """Adapts strategies and methods rates based on adapting_args and on adapting_rate

        Parameters
        ----------
        rates_dict : dict
            Dictionary of rates
        """
        if 'frequency' in self.adapting_args:
            chance_dict = copy.copy(self.improvements_gens_diff)
            chance_dict.sort()
            index = bisect_left(chance_dict, self.gen - self.last_improvement_gen, 0, len(chance_dict))
            difference = len(chance_dict) - index
            proportion_similar = difference / len(chance_dict)
        elif 'fitness_difference' in self.adapting_args:
            chance_dict = copy.copy(self.improvements_diff())
            chance_dict.sort()
            index = bisect_left(chance_dict, self.historical_fitnesses[-1] - self.historical_fitnesses[-2],
                0, len(chance_dict))
            difference = len(chance_dict) - index
            proportion_similar = difference / len(self.historical_fitnesses)
        if 'fitness_increasement' in self.adapting_args:
            rates_dict = self.improvements_increase
        elif 'improvements' in self.adapting_args:
            rates_dict = self.improvements_strategies
        elif 'lineage' in self.adapting_args:
            rates_dict = self.lineage_strategies
        if random.random() < exp(-proportion_similar):
            for n, strategy in enumerate(self.strategies.strategies):
                if self.strategies.rate[n] > 0:
                    self.strategies.rate[n] += self.adapting_rate *\
                        (sum(rates_dict[strategy.__class__.__name__].values()) - self.strategies.rate[n])
                for n, method in enumerate(strategy.methods):
                    if strategy.rate[n] > 0:
                        strategy.rate[n] += self.adapting_rate *\
                            (rates_dict[strategy.__class__.__name__][method.__name__] - strategy.rate[n])

    def local_optimize(self, candidate: Chromosome) -> Genes:
        """Abstract method which must be override in case of local_opt is True for a method which receives a candidate's 
        chromosome and returns its genes locally optimized

        Parameters
        ----------
        candidate : Chromosome
            Candidate which genes are wanted to be optimized

        Returns
        -------
        Genes
            Optimized genes

        Raises
        ------
        Exception
            The local_optimize method must be override
        """
        raise Exception(f'The class {self.__class__.__name__} must override the local_optimize method of parent class')

    def __generate_parent(self, queue: mp.Queue = None, label: str = None) -> Chromosome:
        """Generates a new candidate using the methods from the Create object

        Parameters
        ----------
        queue : mp.Queue, optional
            Multiprocessing queue by which the candidates are returned in case of parallelism, by default None
        label : str, optional
            New parent's chromosome's label, by default None

        Returns
        -------
        Chromosome
            New candidate
        """
        while True:
            try:
                parent = self.create_methods(self.attributes)
                parent.label = label
                parent.lineage = [parent]
                parent.fitness = self.get_fitness(parent)
                break
            except:
                self.catch(parent)
        if queue is not None:
            queue.put(parent)
        return parent

    def __pre_start(self) -> None:
        """Exexcutes at the start of the run method creating the saving directoy and saving the initial parameters in
        config.json
        """
        exclude_list = ('create_methods', 'mutate_methods', 'crossover_methods', 'last_improvement_time', 'gen',
            'historical_fitnesses', 'opt_func', 'start_time', 'attributes', 'best_candidate', 'lineage_ids',
            'strategies', 'improvements_strategies', 'lineage_strategies', 'improvements_gens_diff')
        config_dict = dict()
        config_dict.update({'first_genes': str(self.attributes['first_candidate'].genes)})
        for attrib in inspect.getmembers(self):
            if not attrib[0].startswith('_') and is_jsonable({attrib[0]: attrib[1]}) and not attrib[0] in exclude_list:
                config_dict.update({attrib[0]: attrib[1]})
        for attribute in self.attributes:
            if is_jsonable({attribute: self.attributes[attribute]}) and not attribute in exclude_list:
                config_dict.update({attribute: self.attributes[attribute]})
        config_dict.update({'strategies': {'strategies': {strategy.__class__.__name__:\
            {'methods': [method.__name__ for method in strategy.methods], 'rate': strategy.rate}\
                for strategy in self.strategies.strategies}, 'rate': self.strategies.rate}})
        self.crossover_elitism = [1 for _ in range(self.pool_size)] if self.crossover_elitism is None else\
            self.crossover_elitism
        if self.opt_func == self.__get_improvement_mp:
            config_dict.update({'parallelism': True})
        elif self.opt_func == self.__get_improvement:
            config_dict.update({'parallelism': False})
        if self.save_directory is not None and type(self.save_directory) == str:
            self.save_directory = self.save_directory.rstrip('/') 
        self.save_directory = f"{datetime.now().strftime('%Y_%m_%d_%H:%M')}" if self.save_directory is None \
            else self.save_directory
        os.mkdir(self.save_directory)
        os.mkdir(f'{self.save_directory}/lineage')
        os.mkdir(f'{self.save_directory}/improvements')
        self.attributes['first_candidate'].fitness = self.get_fitness(self.attributes['first_candidate'])
        self.attributes['first_candidate'].lineage = [self.attributes['first_candidate']]
        self.attributes['first_candidate'].label = 'first_candidate'
        self.attributes['best_candidate'] = self.load()
        self.historical_fitnesses.append(self.attributes['best_candidate'].fitness)
        config_dict.update({'first_candidate_fitness': self.attributes['first_candidate'].fitness})
        self.save(self.attributes['first_candidate'], 'first_candidate', self.save_directory)
        with open(self.save_directory + '/config.json', 'w') as file:
            file.write(json.dumps(config_dict, indent=4, sort_keys=True))
        self.start_time = time.time()
        self.last_improvement_gen = 0

    def _generate_end_json(self) -> None:
        """Saves the final state of strategies, improvements_strategies, lineage_strategies and
        improvements_strategies_difference in final.json
        """
        end_dict = {
            'strategies': {'strategies': {strategy.__class__.__name__:\
                {'methods': [method.__name__ for method in strategy.methods], 'rate': strategy.rate}\
                    for strategy in self.strategies.strategies}, 'rate': self.strategies.rate},
            'improvements_strategies': self.improvements_strategies,
            'lineage_strategies': self.lineage_strategies,
            'improvements_gens_diff': self.improvements_gens_diff,
            'improvements_increase': self.improvements_increase,
            'improvements_diff': self.improvements_diff()
        }
        with open(self.save_directory + '/final.json', 'w') as file:
            file.write(json.dumps(end_dict, indent=4, sort_keys=True))

    def run(self) -> Chromosome:
        """Starts the genetic algorithm execution

        Returns
        -------
        Chromosome
            Best candidate
        """
        for timed_out, improvement, time_diff in self.opt_func():
            self.display(improvement, time_diff)
            if timed_out:
                break
        self._generate_end_json()
        return improvement

    def __get_improvement(self) -> Tuple[bool, Chromosome]:
        """Generator of genetic improvements

        Yields
        ------
        Iterator[Tuple[bool, Chromosome]]
            Best candidate achieved until the moment
        """
        self.__pre_start()
        p_time = 0
        yield False, self.attributes['best_candidate'], p_time
        parents = []
        for n in range(self.pool_size):
            if self.max_seconds is not None and p_time >= self.max_seconds:
                yield True, self.attributes['best_candidate'], p_time
            if self.time_toler is not None and p_time - self.last_improvement_time > self.time_toler:
                yield True, self.attributes['best_candidate'], p_time
            if self.max_fitness is not None and parent.fitness >= self.max_fitness:
                 yield True, self.attributes['best_candidate'], p_time
            parent = self.__generate_parent(label=f'{n}_{n}')
            p_time = time.time() - self.start_time
            if parent.fitness > self.attributes['best_candidate'].fitness:
                parent = self.__local_optimization(parent)
                yield False, parent, p_time
                self.last_improvement_time = p_time
                self.attributes['best_candidate'] = parent
                self._write_best()
                self.attributes['best_candidate'].lineage = []
                self.historical_fitnesses.append(parent.fitness)
            parents.append(parent)
        last_parent_index = self.pool_size - 1
        pindex = 1
        n = self.pool_size
        with open(f'{self.save_directory}/time_gen.log', 'a') as file:
            file.write('{0}'.format(time.time() - self.start_time))
            for parent in parents:
                file.write('\t{0}'.format(parent.fitness))
            file.write('\n')
        while True:
            n += 1
            if self.max_seconds is not None and p_time >= self.max_seconds:
                yield True, self.attributes['best_candidate'], p_time
            if self.time_toler is not None and p_time - self.last_improvement_time > self.time_toler:
                yield True, self.attributes['best_candidate'], p_time
            if self.max_fitness is not None and self.attributes['best_candidate'].fitness >= self.max_fitness:
                yield True, self.attributes['best_candidate'], p_time
            pindex = pindex - 1 if pindex > 0 else last_parent_index
            parent = parents[pindex]
            child = self.__get_child(parents, pindex, label=f'{n}_{pindex}')
            p_time = time.time() - self.start_time
            if parent.fitness > child.fitness:
                if self.max_age is None:
                    continue
                parent.age += 1
                if self.max_age > parent.age:
                    continue
                index = bisect_left(self.historical_fitnesses, child.fitness, 0, len(self.historical_fitnesses))
                difference = len(self.historical_fitnesses) - index
                proportion_similar = difference / len(self.historical_fitnesses)
                if random.random() < exp(-proportion_similar):
                    parents[pindex] = child
                    continue
                parents[pindex] = self.attributes['best_candidate']
                parent.age = 0
                continue
            if not child.fitness > parent.fitness:
                child.age = parent.age + 1
                parents[pindex] = child
                continue
            parents[pindex] = child
            parent.age = 0
            if child.fitness > self.attributes['best_candidate'].fitness:
                child = self.__local_optimization(child)
                yield False, child, p_time
                self.last_improvement_time = p_time
                self.attributes['best_candidate'] = child
                self._write_best()
                self.attributes['best_candidate'].lineage = []
                self.historical_fitnesses.append(child.fitness)
                self.last_improvement_gen = self.gen
            with open(f'{self.save_directory}/time_gen.log', 'a') as file:
                file.write('{0}'.format(time.time() - self.start_time))
                for parent in parents:
                    file.write('\t{0}'.format(parent.fitness))
                file.write('\n')
            self.update_parameters()

    def __get_improvement_mp(self) -> Tuple[bool, Chromosome]:
        """Genetic improvements generator using multiprocessing

        Yields
        ------
        Iterator[Tuple[bool, Chromosome]]
            Best candidate achieved until the moment

        Raises
        ------
        Exception
            Elitism rate is incompatible with pool size
        """
        self.__pre_start()
        elit_size = len(self.elitism_rate) if self.elitism_rate is not None else None
        self.last_improvement_time = self.start_time
        queue = mp.Queue(maxsize=self.pool_size)
        processes = []
        self.gen = 0
        if elit_size is not None:
            if sum(self.elitism_rate) > self.pool_size:
                raise Exception('Elitism exceeds pool size.')
        yield False, self.attributes['best_candidate'], 0
        parents = []
        for n in range(self.pool_size):
            processes.append(mp.Process(target=self.__generate_parent, args=(queue, f'{self.gen}_{n}')))
        for process in processes:
            process.start()
        for _ in range(self.pool_size):
            parents.append(queue.get())
        for process in processes:
            process.join()
        gen_time = time.time() - self.start_time
        sorted_next_gen = copy.copy(parents)
        sorted_next_gen.sort(key=lambda c: c.fitness, reverse=False)
        for parent in sorted_next_gen:
            if parent.fitness > self.attributes['best_candidate'].fitness:
                parent = self.__local_optimization(parent)
                yield False, parent, gen_time
                self.attributes['best_candidate'] = parent
                self._write_best(gen_time)
                self.attributes['best_candidate'].lineage = []
                self.last_improvement_time = gen_time
                self.historical_fitnesses.append(parent.fitness)
        parents.sort(key=lambda p: p.fitness, reverse=True)
        with open(f'{self.save_directory}/time_gen.log', 'a') as file:
            file.write('{0}'.format(time.time() - self.start_time))
            for parent in parents:
                file.write('\t{0}'.format(parent.fitness))
            file.write('\n')
        while True:
            self.gen += 1
            if self.max_seconds is not None and time.time() - self.start_time >= self.max_seconds:
                yield True, self.attributes['best_candidate'], gen_time
            if self.max_gens is not None and self.gen >= self.max_gens:
                yield True, self.attributes['best_candidate'], gen_time
            if self.gens_toler is not None and self.gen - self.last_improvement_gen() > self.gens_toler + 1:
                yield True, self.attributes['best_candidate'], gen_time
            if self.time_toler is not None and time.time() - self.last_improvement_time > self.time_toler:
                yield True, self.attributes['best_candidate'], gen_time
            if self.max_fitness is not None and self.attributes['best_candidate'].fitness >= self.max_fitness:
                yield True, self.attributes['best_candidate'], gen_time
            next_gen = []
            queue = mp.Queue(maxsize=self.pool_size)
            processes = []
            results = dict()
            if elit_size is not None:
                for pindex in range(elit_size):
                    for i in range(self.elitism_rate[pindex]):
                        processes.append(mp.Process(target=self.__get_child, args=(parents, pindex, queue, i + \
                            sum(self.elitism_rate[:pindex]), f'{self.gen}_{i + sum(self.elitism_rate[:pindex])}')))
                for pindex in range(elit_size, self.pool_size - sum(self.elitism_rate) + elit_size):
                    processes.append(mp.Process(target=self.__get_child, args=(parents, pindex, queue, 
                        sum(self.elitism_rate) + pindex - elit_size, 
                        f'{self.gen}_{sum(self.elitism_rate) + pindex - elit_size}')))
            else:
                for pindex in range(self.pool_size):
                    processes.append(mp.Process(target=self.__get_child, args=(parents, pindex, queue, pindex,
                    f'{self.gen}_{pindex}')))
            for process in processes:
                process.start()
            for _ in range(self.pool_size):
                results.update(queue.get())
            for process in processes:
                process.join()
            for i in range(self.pool_size):
                next_gen.append(results[i])
            gen_time = time.time() - self.start_time
            sorted_next_gen = copy.copy(next_gen)
            sorted_next_gen.sort(key=lambda c: c.fitness, reverse=False)
            for child in sorted_next_gen:
                if child.fitness > self.attributes['best_candidate'].fitness:
                    child = self.__local_optimization(child)
                    self.attributes['best_candidate'] = child
                    self._write_best(gen_time)
                    yield False, child, gen_time
                    self.attributes['best_candidate'].lineage = []
                    self.historical_fitnesses.append(child.fitness)
                    self.last_improvement_time = gen_time
                    self.last_improvement_gen = self.gen
            for pindex in range(self.pool_size):
                if next_gen[pindex].fitness < parents[pindex].fitness:
                    if self.max_age is None:
                        continue
                    parents[pindex].age += 1
                    if parents[pindex].age < self.max_age:
                        continue
                    index = bisect_left(self.historical_fitnesses, next_gen[pindex].fitness, 0,
                        len(self.historical_fitnesses))
                    difference = len(self.historical_fitnesses) - index
                    proportion_similar = difference / len(self.historical_fitnesses)
                    if random.random() < exp(-proportion_similar):
                        next_gen[pindex].age = parents[pindex].age
                        parents[pindex] = next_gen[pindex]
                        continue
                    parents[pindex] = copy.deepcopy(self.attributes['best_candidate'])
                    parents[pindex].age = 0
                    continue
                if not next_gen[pindex].fitness > parents[pindex].fitness:
                    next_gen[pindex].age = parents[pindex].age + 1
                    parents[pindex] = next_gen[pindex]
                    continue
                parents[pindex] = next_gen[pindex]
                parents.sort(key=lambda p: p.fitness, reverse=True)
            with open(f'{self.save_directory}/time_gen.log', 'a') as file:
                file.write('{0}'.format(time.time() - self.start_time))
                for parent in parents:
                    file.write('\t{0}'.format(parent.fitness))
                file.write('\n')
            self.update_parameters()

    def _write_best(self, gen_time: float) -> None:
        """Registers the data of the best candidate

        Parameters
        ----------
        gen_time : float
            The execution time the generation's fitnesses were calculated
        """
        if self.mutate_after_crossover:
            best_candidate_strategy = copy.copy(self.attributes['best_candidate'].strategy)
            for n, method in enumerate(self.attributes['best_candidate'].strategy):
                if method in self.crossover_methods.methods:
                    best_candidate_strategy.remove(self.attributes['best_candidate'].strategy[n+1])
        for method in best_candidate_strategy:
            for strategy in self.improvements_increase:
                if method.__name__ in self.improvements_increase[strategy]:
                    self.improvements_increase[strategy][method.__name__] +=\
                        (best_candidate_strategy.count(method)/\
                            len(best_candidate_strategy)) *\
                                (self.attributes['best_candidate'].fitness - self.historical_fitnesses[-1])
                    break
        self.improvements_gens_diff.append(self.gen - self.last_improvement_gen)
        self.save(self.attributes['best_candidate'],
            f"{len(self.historical_fitnesses) -1}_{self.attributes['best_candidate'].label}",
            f'{self.save_directory}/improvements')
        for method in best_candidate_strategy:
            for strategy in self.improvements_strategies:
                if method.__name__ in self.improvements_strategies[strategy]:
                    self.improvements_strategies[strategy][method.__name__] +=\
                        best_candidate_strategy.count(method)
                    break
        with open(f'{self.save_directory}/improvements_strategies.log', 'a') as islog:
            islog.write(f"{self.attributes['best_candidate'].strategy_str}\t\
{self.attributes['best_candidate'].fitness}\t{gen_time}\n")
        with open(f'{self.save_directory}/lineage_strategies.log', 'a') as lslog:
            for ancestor in self.attributes['best_candidate'].lineage:
                if not ancestor.label in self.lineage_ids:
                    lslog.write(f'{ancestor.strategy_str}\t{ancestor.fitness}\t{gen_time}\n')
                    self.save(ancestor, f'{len(self.lineage_ids)}_{ancestor.label}', f'{self.save_directory}/lineage')
                    self.lineage_ids.append(ancestor.label)
                    for method in ancestor.strategy:
                        for strategy in self.lineage_strategies:
                            if method.__name__ in self.lineage_strategies[strategy]:
                                self.lineage_strategies[strategy][method.__name__] +=\
                                    self.attributes['best_candidate'].strategy.count(method)
                                break


def is_jsonable(obj: Any) -> bool:
    """Verifies if an object is can be serialized to a JSON formatted str

    Parameters
    ----------
    obj : Any
        Object which will be verified

    Returns
    -------
    bool
        True if obj can be serialized to a JSON formatted str, otherwise False
    """
    try:
        json.dumps(obj)
        return True
    except:
        return False
