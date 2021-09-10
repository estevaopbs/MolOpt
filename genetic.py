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
