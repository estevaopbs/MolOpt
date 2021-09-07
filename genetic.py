from enum import Enum
import Molecular


class Chromosome:
    Genes = None
    Fitness = None
    Age = 0
    Strategy_1 = None
    Strategy_2 = None

    def __init__(self, genes, fitness, strategy1, strategy2):
        self.Genes = genes
        self.Fitness = fitness
        self.Strategy_1 = strategy1
        self.Strategy_2 = strategy2


class Strategies(Enum):
    Create = 0
    Mutate = 1
    Crossover = 2


class Create(Enum):
    full_random = 0
    from_file = 1
    random_from_file = 2


class Mutate(Enum):
    swap_mutate = 0
    mutate_angles = 1
    mutate_distances = 2


class Crossover(Enum):
    crossover_n = 0
    crossover_1 = 1
    crossover_2 = 2
