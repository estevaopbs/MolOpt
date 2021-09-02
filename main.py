from Molecular import *
from genetical import genetic
import unittest
import datetime


def create_random():
    pass


def create_from_file():
    pass


class Optimization_tests(unittest.TestCase):
    def test_h2co(self):
        population = 20
        elitism_rate = 0.2      # must be between 0 and 1
        crossover_rate = 0.2    # must be between 0 and 1
        mutate_methods = []
        file0 = ''