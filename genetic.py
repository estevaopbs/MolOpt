import time
from bisect import bisect_left
from math import exp
from molecular import *
import multiprocessing as mp


class Chromosome:
    def __init__(self, genes=None, fitness=None, strategy=None, method=None, Age=0):
        self.Genes = genes
        self.Fitness = fitness
        self.Strategy = strategy
        self.Method = method
        self.Age = Age


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


def display(candidate, start_time):
    time_diff = time.time() - start_time
    print("{0}\t{1}".format((candidate.Fitness), str(time_diff)))


def get_fitness(genes, fitness_param, threads_per_calculation):
    return - float(genes.get_value([fitness_param], nthreads=threads_per_calculation)[fitness_param])


def get_improvement(new_child, first_parent, generate_parent, maxAge, poolSize, maxSeconds=None, time_toler=None):
    startTime = time.time()
    bestParent = first_parent
    last_improvement_time = startTime
    yield maxSeconds is not None and time.time() - startTime > maxSeconds, bestParent
    parents = [bestParent]
    historicalFitnesses = [bestParent.Fitness]
    for n in range(poolSize - 1):
        parent = generate_parent(label=f'{n+1}')
        if maxSeconds is not None and time.time() - startTime > maxSeconds:
            yield True, parent
        if parent.Fitness > bestParent.Fitness:
            yield False, parent
            last_improvement_time = time.time()
            bestParent = parent
            historicalFitnesses.append(parent.Fitness)
        parents.append(parent)
    lastParentIndex = poolSize - 1
    pindex = 1
    n = poolSize
    while True:
        n += 1
        if maxSeconds is not None and time.time() - startTime > maxSeconds:
            yield True, bestParent
        if time_toler is not None and time.time() - last_improvement_time > time_toler:
            yield True, bestParent
        pindex = pindex - 1 if pindex > 0 else lastParentIndex
        parent = parents[pindex]
        child = new_child(parents, pindex, label=f'{n}')
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
            last_improvement_time = time.time()
            bestParent = child
            historicalFitnesses.append(child.Fitness)


def get_improvement_mp(new_child, first_parent, generate_parent, maxAge, poolSize, maxSeconds, elit_size, elitism_rate,
    max_gen, gen_toler, time_toler):
    startTime = time.time()
    last_improvement_time = startTime
    queue = mp.Queue(maxsize=poolSize - 1)
    processes = []
    gen = 0
    if elit_size is not None:
        if elitism_rate is None:
            elitism_rate = [2 for _ in range(elit_size)]
            if sum(elitism_rate) > poolSize:
                raise Exception('Minimal elitism exceeds pool size. Increase the pool size or reduce the elit size.')
    bestParent = first_parent
    yield maxSeconds is not None and time.time() - startTime > maxSeconds, bestParent
    parents = [bestParent]
    historicalFitnesses = [bestParent.Fitness]
    for n in range(poolSize - 1):
        processes.append(mp.Process(target=generate_parent, args=(queue, f'{gen}_{n+1}')))
    for process in processes:
        process.start()
    for _ in range(poolSize - 1):
        parents.append(queue.get())
    for process in processes:
        process.join()
    sorted_next_gen = copy.copy(parents)
    sorted_next_gen.sort(key=lambda c: c.Fitness, reverse=False)
    for parent in sorted_next_gen:     
        if parent.Fitness > bestParent.Fitness:
            yield False, parent
            bestParent = parent
            last_improvement_time = time.time()
            historicalFitnesses.append(parent.Fitness)
    parents.sort(key=lambda p: p.Fitness, reverse=True)
    last_improvement_gen = 0
    while True:
        gen += 1
        if maxSeconds is not None and time.time() - startTime > maxSeconds:
            yield True, bestParent
        if max_gen is not None and gen > max_gen:
            yield True, bestParent
        if gen_toler is not None and gen - last_improvement_gen > gen_toler + 1:
            yield True, bestParent
        if time_toler is not None and time.time() - last_improvement_time > time_toler:
            yield True, bestParent
        next_gen = []
        queue = mp.Queue(maxsize=poolSize)
        processes = []
        results = dict()
        if elit_size is not None:
            for pindex in range(elit_size):
                for i in range(elitism_rate[pindex]):
                    processes.append(mp.Process(target=new_child, args=(parents, pindex, queue, pindex + i, f'{gen}_{i + sum(elitism_rate[:pindex])}')))
            for pindex in range(elit_size, poolSize - sum(elitism_rate) + elit_size):
                processes.append(mp.Process(target=new_child, args=(parents, pindex, queue, pindex, f'{gen}_{sum(elitism_rate) + pindex - elit_size}')))
        else:
            for pindex in range(poolSize):
                processes.append(mp.Process(target=new_child, args=(parents, pindex, queue, pindex, f'{pindex}')))
        for process in processes:
            process.start()
        for _ in range(poolSize):
            results.update(queue.get())
        for process in processes:
            process.join()
        for i in range(poolSize):
            next_gen.append(results[i])
        sorted_next_gen = copy.copy(next_gen)
        sorted_next_gen.sort(key=lambda c: c.Fitness, reverse=False)
        for child in sorted_next_gen:
            if child.Fitness > bestParent.Fitness:
                yield False, child
                bestParent = child
                historicalFitnesses.append(child.Fitness)
                last_improvement_gen = gen
                last_improvement_time = time.time()
        for pindex in range(poolSize):
            if next_gen[pindex].Fitness < parents[pindex].Fitness:
                if maxAge is None:
                    continue
                parents[pindex].Age += 1
                if parents[pindex].Age < maxAge:
                    continue
                index = bisect_left(historicalFitnesses, next_gen[pindex].Fitness, 0, len(historicalFitnesses))
                difference = len(historicalFitnesses) - index
                proportionSimilar = difference / len(historicalFitnesses)
                if random.random() < exp(-proportionSimilar):
                    next_gen[pindex].Age = parents[pindex].Age
                    parents[pindex] = next_gen[pindex]
                    continue
                parents[pindex] = copy.deepcopy(bestParent)
                parents[pindex].Age = 0
                continue
            if not next_gen[pindex].Fitness > parents[pindex].Fitness:
                next_gen[pindex].Age = parents[pindex].Age + 1
                parents[pindex] = next_gen[pindex]
                continue
            parents[pindex] = next_gen[pindex]
            parents.sort(key=lambda p: p.Fitness, reverse=True)


def optimize(first_molecule:Molecule, fitness_param:str, strategies, max_age:int=None, pool_size:int=1, 
    max_seconds:float=None, time_tolerance:int=None, crossover_elitism=lambda x: 1, 
    mutate_after_crossover:bool=False, parallelism:bool=False, elit_size:int=0, elitism_rate:list=None, 
    generations_tolerance:int=None, threads_per_calculation:int=1, max_gens=None, mutation_rate=1):

    start_time= time.time()

    for strategy in strategies.strategies:
        if type(strategy) is Mutate:
            mutate_methods = strategy
        elif type(strategy) is Create:
            create_methods = strategy
        elif type(strategy) is Crossover:
            crossover_methods = strategy

    mutate_lookup = {
        Mutate.swap_mutate: lambda p, d=0: swap_mutate(p, mutation_rate),
        Mutate.mutate_angles: lambda p, d=0: mutate_angles(p, mutation_rate),
        Mutate.mutate_distances: lambda p, d=0: mutate_distances(p, mutation_rate)
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
    }
    strategy_lookup = {
        Strategies.Create: create_lookup,
        Strategies.Mutate: mutate_lookup,
        Strategies.Crossover: crossover_lookup
    }

    def get_child(candidates, parent_index, queue:mp.Queue=None, child_index:int=None, label:str=None):
        while True:
            try:
                parent = candidates[parent_index]
                sorted_candidates = copy.copy(candidates)
                sorted_candidates.sort(reverse=True, key=lambda p: p.Fitness)
                donor = random.choices(sorted_candidates, [crossover_elitism(n) for n in\
                    reversed(range(len(sorted_candidates)))])[0]
                child = Chromosome()
                child.Strategy = random.choices(strategies.strategies, strategies.rate)[0]
                child.Method = random.choices(child.Strategy.methods, child.Strategy.rate)[0]
                child.Genes = strategy_lookup[child.Strategy.strategy][child.Method](parent.Genes, donor.Genes)
                child.Genes.label = label
                child.Fitness = get_fitness(child.Genes, fitness_param, threads_per_calculation)
                break
            except:
                os.remove(f'data/{child.Genes.label}.inp')
                os.remove(f'data/{child.Genes.label}.out')
                os.remove(f'data/{child.Genes.label}.xml')
                continue
        if queue is not None:
            queue.put({child_index: child})
        return child

    def fn_generate_parent(queue=None, label:str=None):
        while True:
            try:
                parent = Chromosome()
                parent.Genes = create_lookup[random.choices(create_methods.methods, create_methods.rate)[0]](first_molecule)
                parent.Genes.label = label
                parent.Fitness = get_fitness(parent.Genes, fitness_param, threads_per_calculation)
                break
            except:
                os.remove(f'data/{parent.Genes.label}.inp')
                os.remove(f'data/{parent.Genes.label}.out')
                os.remove(f'data/{parent.Genes.label}.xml')
                continue
        if queue is not None:
            queue.put(parent)
        return parent

    strategies_log = open('strategies_log.txt', 'a+')
    usedStrategies = []
    if not parallelism:
        first_molecule.label = '0'
    else:
        first_molecule.label = '0_0'
    first_parent = Chromosome(first_molecule, get_fitness(first_molecule, fitness_param, threads_per_calculation), 
    None, None)
    n = 0
    if not parallelism:
        for timedOut, improvement in get_improvement(get_child, first_parent, fn_generate_parent, max_age, pool_size, 
        max_seconds, time_tolerance):
            improvement.Genes.save(f'{n}_{improvement.Genes.label}', 'improvements')
            display(improvement, start_time)
            f = (improvement.Strategy, improvement.Method)
            strategies_log.write(str(f).replace('(', '').replace(')', '') + '\n')
            usedStrategies.append(f)
            n += 1
            if timedOut:
                break
    else:
        for timedOut, improvement in get_improvement_mp(get_child, first_parent, fn_generate_parent, max_age, pool_size,
        max_seconds, elit_size, elitism_rate, max_gens, generations_tolerance, time_tolerance):
            improvement.Genes.save(f'{n}_{improvement.Genes.label}', 'improvements')
            display(improvement, start_time)
            f = (improvement.Strategy, improvement.Method)
            strategies_log.write(str(f).replace('(', '').replace(')', '') + '\n')
            usedStrategies.append(f)
            n += 1
            if timedOut:
                break
    return improvement.Genes, usedStrategies
