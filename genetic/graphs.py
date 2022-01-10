from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib
import tikzplotlib
from numpy import mean
from itertools import product
import os
from typing import Tuple
matplotlib.rcParams['figure.max_open_warning'] = 24


"""Graphs production for genetic outputs
"""


class Data:
    """Reads the data from the genetic output folder and produces graphs using matplotlib
    """
    def __init__(self, data_dir: str, save_dir: str = 'graphs') -> Data:
        """Initiates the object by receiving the directory created by the Genetic object

        :param data_dir: Directory created by the Genetic object which contains all saved data
        :type data_dir: str
        :param save_dir: Directory in which the graphs will be saved, defaults to None
        :type save_dir: str
        """
        self.data_dir = data_dir
        self.save_dir = save_dir
        with open(f'{data_dir}/time_gen.log', 'r') as file:
            self.generations = [[float(n) for n in row] for row in [item.split('\t')\
                for item in file.read().splitlines()]]
        with open(f'{data_dir}/improvements_strategies.log', 'r') as file:
            self.improvements = [[[sstr for sstr in row[0].replace('[', '').replace(']', '').split(', ')], 
                float(row[1]), float(row[2])] for row in [item.split('\t') for item in file.read().splitlines()]]
        with open(f'{data_dir}/lineage_strategies.log', 'r') as file:
            self.lineage = [[[sstr for sstr in row[0].replace('[', '').replace(']', '').split(', ')], 
                float(row[1]), float(row[2])] for row in [item.split('\t') for item in file.read().splitlines()]]
        self.improvements_files = [item[:-4] for item in os.listdir(f'{data_dir}/improvements')]
        self.improvements_files.sort(key=lambda x: int(x.split('_')[0]))
        self.lineage_files = [item[:-4] for item in os.listdir(f'{data_dir}/lineage')]
        self.lineage_files.sort(key=lambda x: int(x.split('_')[1]))

    @staticmethod
    def ymod(y_value: float) -> float:
        """In case of the fitness used in genetic be actually a function of the wanted parameter, here you can do the
        opposite way by overriding this method. This method receives all the fitness values and returns to the plot
        function the value converted to the way it will appear in the graph. By default, it means, if this methods don't
        be override the y axis in all the graphs (excluding the strategies ones) will correspond to the fitness

        :param y_value: Fitness value
        :type y_value: float
        :return: Converted value
        :rtype: float
        """
        return y_value

    def plot(self, *args: Tuple[str, str], format: str, fit_str: str = 'Fitness', mean_fit_str: str = 'Medium fitness',
        time_str: str = 'Time', gen_str: str = 'Generation') -> None:
        """Produces the plots by receiving Tuples of with a the string correspondent to the paramter x and another
        correspondent to the parameter y. The x parameter strings can be: time; generation; n. The y parameter strings
        can be: best; mean; lineage; lineage_mean; improvement; lineage_best; improvements_strategies;
        lineage_strategies

        :param format: The format the graphs will be saved. It can be: tex; png; eps; jpeg; jpg; pdf; pgf; ps; raw; 
            rgba; svg; svgz; tif; tiff
        :type format: str, optional
        :param fit_str: String that will describe the y axis in the graphs (excluding the strategies ones, and the mean
            ones), defaults to 'Fitness'
        :type fit_str: str, optional
        :param mean_fit_str: String that will describe the y axis in the graphs of mean values, defaults to
            'Medium fitness'
        :type mean_fit_str: str, optional
        :param time_str: String which will label the time axes, defaults to 'Time'
        :type time_str: str
        :param gen_str: String which will label the generation axes, defaults to 'Generation'
        :type gen_str: str
        :raises Exception: Raises an exception if paramter x is not valid
        :raises Exception: Raises an exception if paramter y is not valid
        """
        xparams_args = ('time', 'generation', 'n')
        yparams_args = ('best', 'mean', 'lineage', 'lineage_mean', 'improvement', 'lineage_best',
            'improvements_strategies', 'lineage_strategies')
        os.mkdir(self.save_dir)
        if args == ('all',):
            args = product(xparams_args, yparams_args)
        for arg in args:
            xparam, yparam = arg
            if not yparam in yparams_args:
                raise Exception(f'{yparam} is not valid')
            if not xparam in xparams_args:
                raise Exception(f'{xparam} is not valid')
            fig, ax = plt.subplots()
            if yparam == 'best':
                ydata = [self.ymod(gen[1]) for gen in self.generations]
            elif yparam == 'mean':
                ydata = [self.ymod(mean(gen[1:])) for gen in self.generations]
            elif yparam == 'improvement':
                ydata = [self.ymod(i[1]) for i in self.improvements]
                if xparam == 'time':
                    xdata = [i[2] for i in self.improvements]
                elif xparam == 'generation':
                    xdata = [i.split('_')[1] for i in self.improvements_files]
                elif xparam == 'n':
                    pass
                else:
                    raise Exception(f'{xparam} is not valid')
            elif yparam == 'lineage':
                ydata = [self.ymod(self.generations[int(ancestor_label.split('_')[1])]\
                    [int(ancestor_label.split('_')[2]) + 1]) for ancestor_label in self.lineage_files]
                if xparam == 'time':
                    xdata = [self.generations[int(ancestor_label.split('_')[1])][0]\
                        for ancestor_label in self.lineage_files]
                elif xparam == 'generation':
                    xdata = [int(ancestor_label.split('_')[1]) for ancestor_label in self.lineage_files]
                elif xparam == 'n':
                    pass
                else:
                    raise Exception(f'{xparam} is not valid')
            elif yparam == 'lineage_mean':
                ydatagens = list({int(ancestor_label.split('_')[1]) for ancestor_label in self.lineage_files})
                ydatagens.sort()
                ydata = [mean([self.ymod(self.generations[n][int(ancestor_label.split('_')[2]) + 1])\
                    for ancestor_label in self.lineage_files if int(ancestor_label.split('_')[1]) == n])\
                        for n in ydatagens]
                if xparam == 'time':
                    xdata = [self.generations[n][0] for n in ydatagens]
                elif xparam == 'generation':
                    xdata = ydatagens
                elif xparam == 'n':
                    pass
                else:
                    raise Exception(f'{xparam} is not valid')
            elif yparam == 'lineage_best':
                ydatagens = list({int(ancestor_label.split('_')[1]) for ancestor_label in self.lineage_files})
                ydatagens.sort()
                ydata = [self.ymod(max([self.generations[n][int(ancestor_label.split('_')[2]) + 1]\
                    for ancestor_label in self.lineage_files if int(ancestor_label.split('_')[1]) == n]))\
                        for n in ydatagens]
                if xparam == 'time':
                    xdata = [self.generations[n][0] for n in ydatagens]
                elif xparam == 'generation':
                    xdata = ydatagens
                elif xparam == 'n':
                    pass
                else:
                    raise Exception(f'{xparam} is not valid')
            elif yparam == 'lineage_strategies':
                strategies_data = dict()
                for n, ancestor in enumerate(self.lineage_files):
                    ancestor_strategies = self.lineage[int(ancestor.split('_')[0])][0]
                    for strategy in set(ancestor_strategies):
                        if strategy in strategies_data.keys():
                            strategies_data[strategy].append(strategies_data[strategy][-1] +\
                                ancestor_strategies.count(strategy))
                        else:
                            strategies_data.update({strategy: [0 for _ in range(n)] +\
                                [ancestor_strategies.count(strategy)]})
                    for strategy in strategies_data.keys():
                        if not strategy in ancestor_strategies:
                            strategies_data[strategy].append(strategies_data[strategy][-1])
                if xparam == 'time':
                    xdata = [self.generations[int(ancestor.split('_')[1])][0] for ancestor in self.lineage_files]
                elif xparam == 'generation':
                    xdata = [int(ancestor_label.split('_')[1]) for ancestor_label in self.lineage_files]
                elif xparam == 'n':
                    pass
                else:
                    raise Exception(f'{xparam} is not valid')
            elif yparam == 'improvements_strategies':
                strategies_data = dict()
                for n, improvement in enumerate(self.improvements):
                    for strategy in set(improvement[0]):
                        if strategy in strategies_data.keys():
                            strategies_data[strategy].append(strategies_data[strategy][-1] +\
                                improvement[0].count(strategy))
                        else:
                            strategies_data.update({strategy: [0 for _ in range(n)] + [improvement[0].count(strategy)]})
                    for strategy in strategies_data.keys():
                        if not strategy in improvement[0]:
                            strategies_data[strategy].append(strategies_data[strategy][-1])
                if xparam == 'time':
                    xdata = [improvement[2] for improvement in self.improvements]
                elif xparam == 'generation':
                    xdata = [int(i.split('_')[1]) for i in self.improvements_files]
                elif xparam == 'n':
                    xdata = list(range(len(self.improvements)))
                else:
                    raise Exception(f'{xparam} is not valid')
            else:
                raise Exception(f'{yparam} is not valid')
            if yparam == 'best' or yparam == 'mean':
                if xparam == 'time':
                    xdata = [gen[0] for gen in self.generations]
                elif xparam == 'generation':
                    xdata = range(len(self.generations))
                elif xparam == 'n':
                    pass
                else:
                    raise Exception(f'{xparam} is not valid')
            if 'strategies' in yparam and xparam != 'n':
                if xparam == 'time':
                    plt.xlabel(rf'{time_str} ($s$)')
                elif xparam == 'generation':
                    plt.xlabel(f'{gen_str}')
                else:
                    raise Exception(f'{xparam} is not valid')
                for strategy in strategies_data.keys():
                    ax.plot(xdata, strategies_data[strategy], drawstyle='steps', label=strategy)
                plt.legend()
                if format == 'tex':
                    tikzplotlib.save(f'{self.save_dir}/{xparam}_{yparam}.tex')
                else:
                    plt.savefig(f'{self.save_dir}/{xparam}_{yparam}.{format}')
                print(f'{self.data_dir} data x={xparam} y={yparam} plotted with success')
            elif not 'strategies' in yparam:
                if 'mean' in yparam:
                    plt.ylabel(mean_fit_str)
                else:
                    plt.ylabel(fit_str)
                if xparam == 'time':
                    plt.xlabel(rf'{time_str} ($s$)')
                if xparam == 'generation':
                    plt.xlabel(f'{gen_str}')
                if xparam == 'n':
                    xdata = list(range(len(ydata)))
                    plt.xlabel(r'$N$')
                ax.plot(xdata, ydata, drawstyle='steps')
                if format == 'tex':
                    tikzplotlib.save(f'{self.save_dir}/{xparam}_{yparam}.tex')
                else:
                    plt.savefig(f'{self.save_dir}/{xparam}_{yparam}.{format}')
                print(f'{self.data_dir} data x={xparam} y={yparam} plotted with success')
