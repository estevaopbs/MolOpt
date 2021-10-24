import time
from bisect import bisect_left
from math import exp
from molecular import *
import multiprocessing as mp


class Chromosome:
    __slots__ = ('Genes', 'Fitness', 'Strategy', 'Age', 'Lineage')

    def __init__(self, genes=None, fitness=None, strategy=[], Age=0, Lineage=[]):
        self.Genes = genes
        self.Fitness = fitness
        self.Strategy = strategy
        self.Age = Age
        self.Lineage = Lineage

    @property
    def strategy_str(self):
        return str([[strategy[0].__class__.__name__, strategy[0].log_dict[strategy[1]]] for strategy in self.Strategy])\
            [1:-1].replace("'", "")


class Strategies:
    Create = 0
    Mutate = 1
    Crossover = 2
    __slots__ = ('strategies', 'rate')

    def __init__(self, strategies:list, strategies_rate:list):
        self.strategies = strategies
        self.rate = strategies_rate


class OPTG:
    log_dict = {0: ''}


class Load:
    log_dict = {0: ''}


class Create:
    randomize = 0
    mutate_first = 1
    strategy = 0
    log_dict = {0: 'randomize', 1: 'mutate_first'}
    __slots__ = ('methods', 'rate')

    def __init__(self, methods:list, methods_rate:list):
        self.methods = methods
        self.rate = methods_rate


class Mutate:
    swap_mutate = 0
    mutate_angles = 1
    mutate_distances = 2
    strategy = 1
    log_dict = {0: 'swap_mutate', 1: 'mutate_angles', 2: 'mutate_distances'}
    __slots__ = ('methods', 'rate')
    
    def __init__(self, methods:list, methods_rate:list):
        self.methods = methods
        self.rate = methods_rate


class Crossover:
    crossover_n = 0
    crossover_1 = 1
    crossover_2 = 2
    strategy = 2
    log_dict = {0: 'crossover_n', 1: 'crossover_1', 2: 'crossover_2'}
    __slots__ = ('methods', 'rate')

    def __init__(self, methods:list, methods_rate:list):
        self.methods = methods
        self.rate = methods_rate


def display(candidate, timediff):
    print("{0}\t{1}".format((candidate.Fitness), str(timediff)))


def get_fitness(genes, fitness_param, threads_per_calculation):
    return - float(genes.get_value([fitness_param], nthreads=threads_per_calculation)[fitness_param])


def get_improvement(new_child, first_parent, generate_parent, maxAge, poolSize, fn_optg, maxSeconds, 
    time_toler, use_optg):
    startTime = time.time()
    if use_optg:
        bestParent = fn_optg(first_parent)
    last_improvement_time = startTime
    yield maxSeconds is not None and time.time() - startTime > maxSeconds, bestParent
    bestParent.Lineage = []
    parents = [bestParent]
    historicalFitnesses = [bestParent.Fitness]
    for n in range(poolSize - 1):
        parent = generate_parent(label=f'{n+1}_{n+1}')
        if maxSeconds is not None and time.time() - startTime > maxSeconds:
            yield True, parent
        if time_toler is not None and time.time() - last_improvement_time > time_toler:
            yield True, bestParent
        if parent.Fitness > bestParent.Fitness:
            if use_optg:
                parent = fn_optg(parent)
            yield False, parent
            last_improvement_time = time.time()
            bestParent = parent
            bestParent.Lineage = []
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
        child = new_child(parents, pindex, label=f'{n}_{pindex}', is_parent_best=parent == bestParent)
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
            if use_optg:
                child = fn_optg(child)
            yield False, child
            last_improvement_time = time.time()
            bestParent = child
            bestParent.Lineage = []
            historicalFitnesses.append(child.Fitness)


def get_improvement_mp(new_child, first_parent, generate_parent, maxAge, poolSize, maxSeconds, elit_size, elitism_rate,
    max_gen, gen_toler, time_toler, fn_optg, use_optg):
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
    bestParent = fn_optg(first_parent) if use_optg else first_parent
    yield maxSeconds is not None and time.time() - startTime > maxSeconds, bestParent
    bestParent.Lineage = []
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
            if use_optg:
                parent = fn_optg(parent)
            yield False, parent
            bestParent = parent
            bestParent.Lineage = []
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
                    processes.append(mp.Process(target=new_child, args=(parents, pindex, queue, i + \
                        sum(elitism_rate[:pindex]), f'{gen}_{i + sum(elitism_rate[:pindex])}', parent == bestParent)))
            for pindex in range(elit_size, poolSize - sum(elitism_rate) + elit_size):
                processes.append(mp.Process(target=new_child, args=(parents, pindex, queue, sum(elitism_rate) + pindex \
                    - elit_size, f'{gen}_{sum(elitism_rate) + pindex - elit_size}', parent == bestParent)))
        else:
            for pindex in range(poolSize):
                processes.append(mp.Process(target=new_child, args=(parents, pindex, queue, pindex, f'{gen}_{pindex}',
                    parent == bestParent)))
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
                if use_optg:
                    child = fn_optg(child)
                yield False, child
                bestParent = child
                bestParent.Lineage = []
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
    generations_tolerance:int=None, threads_per_calculation:int=1, max_gens=None, mutation_rate:int=1, 
    use_optg:bool=True):

    start_time= time.time()
    if crossover_elitism is None:
        crossover_elitism = [1 for _ in range(pool_size)]

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

    def get_child(candidates, parent_index, queue:mp.Queue=None, child_index:int=None, label:str=None, 
        is_parent_best:bool=False):
        while True:
            try:
                parent = candidates[parent_index]
                for _ in range(mutation_rate):
                    sorted_candidates = copy.copy(candidates)
                    sorted_candidates.sort(reverse=True, key=lambda p: p.Fitness)
                    donor = random.choices(sorted_candidates, crossover_elitism)[0]
                    child = Chromosome()
                    child.Strategy.append([random.choices(strategies.strategies, strategies.rate)[0]])
                    child.Strategy[-1].append(random.choices(child.Strategy[-1][0].methods, 
                        child.Strategy[-1][0].rate)[0])
                    child.Genes = strategy_lookup[child.Strategy[-1][0].strategy][child.Strategy[-1][1]]\
                        (parent.Genes, donor.Genes)
                    parent = child
                child.Genes.label = label
                child.Fitness = get_fitness(child.Genes, fitness_param, threads_per_calculation)
                child.Lineage = parent.Lineage
                if not is_parent_best:
                    child.Lineage.append(parent)
                break
            except:
                os.remove(f'data/{child.Genes.label}.inp')
                os.remove(f'data/{child.Genes.label}.out')
                os.remove(f'data/{child.Genes.label}.xml')
                continue
        if queue is not None:
            queue.put({child_index: child})
        return child

    def fn_optg(candidate:Chromosome) -> Chromosome:
        new_genes = optg(candidate.Genes, fitness_param, 'data', threads_per_calculation * pool_size)
        new_fitness = -float(new_genes.output_values[fitness_param])
        return Chromosome(new_genes, new_fitness, candidate.Strategy + [[OPTG(), 0]], 0, 
            candidate.Lineage + [candidate])

    def fn_generate_parent(queue=None, label:str=None):
        while True:
            try:
                parent = Chromosome()
                parent.Strategy = [create_methods, random.choices(create_methods.methods, create_methods.rate)[0]]
                parent.Genes = create_lookup[parent.Strategy[1]](first_molecule)
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

    usedStrategies = []
    first_molecule.label = '0_0'
    first_parent = Chromosome(first_molecule, get_fitness(first_molecule, fitness_param, 
        threads_per_calculation * pool_size), [[Load(), 0]])
    if not os.path.exists('lineage'):
        os.mkdir('lineage')
    n = 0
    j = 0
    if not parallelism:
        for timedOut, improvement in get_improvement(get_child, first_parent, fn_generate_parent, max_age, pool_size, 
            fn_optg, max_seconds, time_tolerance, use_optg):
            improvement.Genes.save(f'{n}_{improvement.Genes.label}', 'improvements')
            improvement.Lineage.append(improvement)
            timediff = time.time() - start_time
            display(improvement, timediff)
            with open('improvements_strategies.log', 'a') as islog:
                islog.write(f'{improvement.strategy_str}\t{improvement.Fitness}\t{timediff}\n')
            with open('lineage_strategies.log', 'a') as lslog:
                for ancestor in improvement.Lineage:
                    lslog.write(f'{ancestor.strategy_str}\t{ancestor.Fitness}\t{timediff}\n')
                    ancestor.Genes.save(f'{j}_{ancestor.Genes.label}', 'lineage')
                    j += 1
            usedStrategies.append(improvement.Strategy)
            n += 1
            if timedOut:
                break
    else:
        for timedOut, improvement in get_improvement_mp(get_child, first_parent, fn_generate_parent, max_age, pool_size,
            max_seconds, elit_size, elitism_rate, max_gens, generations_tolerance, time_tolerance, fn_optg,
            use_optg):
            improvement.Genes.save(f'{n}_{improvement.Genes.label}', 'improvements')
            improvement.Lineage.append(improvement)
            timediff = time.time() - start_time
            display(improvement, timediff)
            with open('improvements_strategies.log', 'a') as islog:
                islog.write(f'{improvement.strategy_str}\t{improvement.Fitness}\t{timediff}\n')
            with open('lineage_strategies.log', 'a') as lslog:
                for ancestor in improvement.Lineage:
                    lslog.write(f'{ancestor.strategy_str}\t{ancestor.Fitness}\t{timediff}\n')
                    ancestor.Genes.save(f'{j}_{ancestor.Genes.label}', 'lineage')
                    j += 1
            usedStrategies.append(improvement.Strategy)
            n += 1
            if timedOut:
                break
    improvement.Genes.save(f'{j}_{ancestor.Genes.label}', 'lineage')
    with open('strategies_log.txt', 'a') as slog:
        slog.write('\n---\n\n')
    return improvement.Genes, usedStrategies
