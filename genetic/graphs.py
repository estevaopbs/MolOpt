from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib
import tikzplotlib
from numpy import mean
from itertools import product
import os
from typing import Tuple
import shutil
import json


"""Graphs production for genetic outputs
"""


matplotlib.rcParams['figure.max_open_warning'] = 24


class Data:
    """Read and plot graphs with the data saved by Genetic objects
    """
    __slots__ = ('data_dir', 'generations', 'improvements', 'lineage', 'improvements_files', 'lineage_files')

    def __init__(self, data_dir: str) -> None:
        """Initiates the Data object

        Parameters
        ----------
        data_dir : str
            Directory created by the Genetic object which contains all saved data
        """
        self.data_dir = data_dir
        with open(f'{data_dir}/time_gen.log', 'r') as file:
            self.generations = [[float(n) for n in line] for line in [item.split('\t')\
                for item in file.read().splitlines()]]
        with open(f'{data_dir}/improvements_strategies.log', 'r') as file:
            self.improvements = [[[sstr for sstr in line[0].replace('[', '').replace(']', '').split(', ')], 
                float(line[1]), float(line[2])] for line in [item.split('\t') for item in file.read().splitlines()]]
        with open(f'{data_dir}/lineage_strategies.log', 'r') as file:
            self.lineage = [[[sstr for sstr in line[0].replace('[', '').replace(']', '').split(', ')], 
                float(line[1]), float(line[2])] for line in [item.split('\t') for item in file.read().splitlines()]]
        self.improvements_files = [item.split('.')[0]  for item in os.listdir(f'{data_dir}/improvements')]
        self.improvements_files.sort(key=lambda x: int(x.split('_')[0]))
        self.lineage_files = [item.split('.')[0] for item in os.listdir(f'{data_dir}/lineage')]
        self.lineage_files.sort(key=lambda x: int(x.split('_')[1]))

    def plot(self, *args: Tuple[str, str] | str, format: str, savedir: str, fit_str: str = 'Fitness',
        mean_fit_str: str = 'Medium fitness', time_str: str = 'Time', gen_str: str = 'Generation',
        gen_range: Tuple(int, int) = (0, -1), strategies_dict: dict = dict(), 
        strtgs_fit_str: str = 'Fitness increasement',ymod: function = lambda x: x) -> None:
        """Produces the graphs

        Parameters
        ----------
        *args : Tuple[str, str] | str
            Tuples of strings correspondents to x and y parameters respectively, if it's all then all the possible
            graphs will be plotted. The possible x parameter strings are time, generation, n, and the possibe y
            parameter strings are best, mean, lineage, lineage_mean, improvement, lineage_best, improvements_strategies,
            lineage_strategies
        format : str
            The format the graphs will be saved. It can be tex, png, eps, jpeg, jpg, pdf, pgf, ps, raw, rgba, svg, svgz,
            tif, tiff
        savedir : str
            Directory in which the graphs will be saved
        fit_str : str, optional
            String that will describe the y axis in the graphs (excluding the strategies ones, and the mean ones), by
            default 'Fitness'
        mean_fit_str : str, optional
            String that will describe the y axis in the graphs of mean values, by default 'Medium fitness'
        time_str : str, optional
            String which will label the time axes, by default 'Time'
        gen_str : str, optional
            String which will label the generation axes, by default 'Generation'
        gen_range : Tuple, optional
            Generation range that will be plotted in the graphs, by default (0, -1)
        strategies_dict : dict, optional
            Dictionary with the strings that will override the names of the strategies functions in the strategies
            graphs, by default dict()
        ymod : function, optional
            Function which will modify each fitness value befor it is plotted in the graphs, by default lambda x: x

        Raises
        ------
        Exception
            Parameter x is not valid
        Exception
            Parameter y is not valid
        """
        xparams_args = ('time', 'generation', 'n')
        yparams_args = ('best', 'mean', 'lineage', 'lineage_mean', 'improvement', 'lineage_best',
            'improvements_strategies', 'lineage_strategies', 'strategies_fitness')
        savedir = savedir.rstrip('/')
        if savedir != '' and type(savedir) == str:
            savedir += '/'
        os.mkdir(savedir)
        with open(f'{self.data_dir}/config.json', 'r') as file:
            config = json.loads(file.read())
        if args == ('all',):
            args = product(xparams_args, yparams_args)
        if gen_range[1] == -1:
            gen_range = (gen_range[0], len(self.generations) - 1)
        time_range = (self.generations[gen_range[0]][0], self.generations[gen_range[1]][0])
        if gen_range[0] == 0:
            time_range = (float(self.improvements[0][2]), time_range[1])
        for arg in args:
            xparam, yparam = arg
            if not yparam in yparams_args:
                raise Exception(f'{yparam} is not valid')
            if not xparam in xparams_args:
                raise Exception(f'{xparam} is not valid')
            fig, ax = plt.subplots()
            if yparam == 'best':
                ydata = [ymod(gen[1]) for gen in self.generations if\
                    (gen_range[0] <= self.generations.index(gen) <= gen_range[1])]
            elif yparam == 'mean':
                ydata = [ymod(mean(gen[1:])) for gen in self.generations if\
                    (gen_range[0] <= self.generations.index(gen) <= gen_range[1])]
            elif yparam == 'improvement':
                ydata = [ymod(i[1]) for i in self.improvements if\
                    (time_range[0] <= i[2] <= time_range[1])]
                if xparam == 'time':
                    xdata = [i[2] for i in self.improvements if\
                        (time_range[0] <= i[2] <= time_range[1])]
                elif xparam == 'generation':
                    xdata = [int(i.split('_')[1]) for i in self.improvements_files if\
                        (gen_range[0] <= int(i.split('_')[1]) <= gen_range[1])]
                elif xparam == 'n':
                    pass
                else:
                    raise Exception(f'{xparam} is not valid')
            elif yparam == 'lineage':
                ydata = [ymod(self.generations[int(ancestor_label.split('_')[1])]\
                    [int(ancestor_label.split('_')[2]) + 1]) for ancestor_label in self.lineage_files if\
                        (gen_range[0] <= int(ancestor_label.split('_')[1]) <= gen_range[1])]
                if xparam == 'time':
                    xdata = [self.generations[int(ancestor_label.split('_')[1])][0] for ancestor_label in\
                        self.lineage_files if (gen_range[0] <= int(ancestor_label.split('_')[1]) <= gen_range[1])]
                elif xparam == 'generation':
                    xdata = [int(ancestor_label.split('_')[1]) for ancestor_label in self.lineage_files if\
                        (gen_range[0] <= int(ancestor_label.split('_')[1]) <= gen_range[1])]
                elif xparam == 'n':
                    pass
                else:
                    raise Exception(f'{xparam} is not valid')
            elif yparam == 'lineage_mean':
                ydatagens = list({int(ancestor_label.split('_')[1]) for ancestor_label in self.lineage_files})
                ydatagens.sort()
                ydata = [mean([ymod(self.generations[n][int(ancestor_label.split('_')[2]) + 1])\
                    for ancestor_label in self.lineage_files if int(ancestor_label.split('_')[1]) == n])\
                        for n in ydatagens if (gen_range[0] <= n <= gen_range[1])]
                if xparam == 'time':
                    xdata = [self.generations[n][0] for n in ydatagens if (gen_range[0] <= n <= gen_range[1])]
                elif xparam == 'generation':
                    xdata = [n for n in ydatagens if (gen_range[0] <= n <= gen_range[1])]
                elif xparam == 'n':
                    pass
                else:
                    raise Exception(f'{xparam} is not valid')
            elif yparam == 'lineage_best':
                ydatagens = list({int(ancestor_label.split('_')[1]) for ancestor_label in self.lineage_files})
                ydatagens.sort()
                ydata = [ymod(max([self.generations[n][int(ancestor_label.split('_')[2]) + 1]\
                    for ancestor_label in self.lineage_files if int(ancestor_label.split('_')[1]) == n]))\
                        for n in ydatagens if (gen_range[0] <= n <= gen_range[1])]
                if xparam == 'time':
                    xdata = [self.generations[n][0] for n in ydatagens if (gen_range[0] <= n <= gen_range[1])]
                elif xparam == 'generation':
                    xdata = [n for n in ydatagens if (gen_range[0] <= n <= gen_range[1])]
                elif xparam == 'n':
                    pass
                else:
                    raise Exception(f'{xparam} is not valid')
            elif yparam == 'lineage_strategies':
                strategies_data = dict()
                lineage_data_files = [ancestor for ancestor in self.lineage_files if\
                    (gen_range[0] <= int(ancestor.split('_')[1]) <= gen_range[1])]
                previous_ancestors = [self.lineage[n] for n in [int(_file.split('_')[0]) for _file in \
                    [ancestor for ancestor in self.lineage_files if (gen_range[0]> int(ancestor.split('_')[1]))]]]
                for n, ancestor in enumerate(lineage_data_files):
                    ancestor_strategies = self.lineage[int(ancestor.split('_')[0])][0]
                    for strategy in set(ancestor_strategies):
                        if strategy in strategies_data.keys():
                            strategies_data[strategy].append(strategies_data[strategy][-1] +\
                                ancestor_strategies.count(strategy))
                        else:
                            if gen_range[0] == 0:
                                strategies_data.update({strategy: [0 for _ in range(n)] +\
                                    [ancestor_strategies.count(strategy)]})
                            else:
                                previous_strategy_count = sum([ancestor[0].count(strategy) for ancestor in\
                                    previous_ancestors])
                                strategies_data.update({strategy: [previous_strategy_count for _ in range(n)] +\
                                    [ancestor_strategies.count(strategy)  + previous_strategy_count]})
                    for strategy in strategies_data.keys():
                        if not strategy in ancestor_strategies:
                            strategies_data[strategy].append(strategies_data[strategy][-1])
                if xparam == 'time':
                    xdata = [self.generations[int(ancestor.split('_')[1])][0] for ancestor in lineage_data_files]
                elif xparam == 'generation':
                    xdata = [int(ancestor_label.split('_')[1]) for ancestor_label in lineage_data_files]
                elif xparam == 'n':
                    pass
                else:
                    raise Exception(f'{xparam} is not valid')
            elif yparam == 'improvements_strategies':
                strategies_data = dict()
                improvements_data = [improvement for improvement in self.improvements if\
                    (time_range[0] <= improvement[2] <= time_range[1])]
                previous_improvements = [self.improvements[n] for n in [int(_file.split('_')[0]) for _file in \
                    [improvement for improvement in self.improvements_files if\
                        (gen_range[0] > int(improvement.split('_')[1]))]]]
                for n, improvement in enumerate(improvements_data):
                    for strategy in set(improvement[0]):
                        if strategy in strategies_data.keys():
                            strategies_data[strategy].append(strategies_data[strategy][-1] +\
                                improvement[0].count(strategy))
                        else:
                            if gen_range[0] == 0:
                                strategies_data.update({strategy: [0 for _ in range(n)] +\
                                    [improvement[0].count(strategy)]})
                            else:
                                previous_strategy_count = sum([ancestor[0].count(strategy) for ancestor in\
                                    previous_improvements])
                                strategies_data.update({strategy: [previous_strategy_count for _ in range(n)] +\
                                    [improvement[0].count(strategy) + previous_strategy_count]})
                    for strategy in strategies_data.keys():
                        if not strategy in improvement[0]:
                            strategies_data[strategy].append(strategies_data[strategy][-1])
                if xparam == 'time':
                    xdata = [improvement[2] for improvement in improvements_data]
                elif xparam == 'generation':
                    xdata = [int(i.split('_')[1]) for i in self.improvements_files if\
                        (gen_range[0] <= int(i.split('_')[1]) <= gen_range[1])]
                elif xparam == 'n':
                    xdata = list(range(len(improvements_data)))
                else:
                    raise Exception(f'{xparam} is not valid')
            elif yparam == 'strategies_fitness':
                strategies_data = dict()
                strategies_data.update({method: [] for method in config['strategies']['strategies']['Create']\
                    ['methods']})
                strategies_data.update({method: [] for method in config['strategies']['strategies']['Mutate']\
                    ['methods']})
                strategies_data.update({method: [] for method in config['strategies']['strategies']['Crossover']\
                    ['methods']})
                improvements_data = [improvement for improvement in self.improvements if\
                    (time_range[0] <= improvement[2] <= time_range[1])]
                previous_improvements = [self.improvements[n] for n in [int(_file.split('_')[0]) for _file in \
                    [improvement for improvement in self.improvements_files if\
                        (gen_range[0] > int(ancestor.split('_')[1]))]]]
                previous_strategy_data = dict()
                previous_strategy_data.update({method: [] for method in config['strategies']['strategies']['Create']\
                    ['methods']})
                previous_strategy_data.update({method: [] for method in config['strategies']['strategies']['Mutate']\
                    ['methods']})
                previous_strategy_data.update({method: [] for method in config['strategies']['strategies']['Crossover']\
                    ['methods']})
                if gen_range[0] > 0:
                    previous_strategy_data[self.improvements[0][0][0]].append(self.improvements[0][1] -\
                        config['first_candidate_fitness'])
                    for method in strategies_data:
                        if method != self.improvements[0][0][0]:
                            previous_strategy_data[method].append(0)
                    for n, improvement in enumerate(previous_improvements[1:]):
                        for strategy in set(improvement[0]):
                            previous_strategy_data[strategy] += (improvement[0].count(strategy)/len(improvement[0])) *\
                                (improvement[1] - self.improvements[n][1])
                elif gen_range[0] == 0:
                    strategies_data[self.improvements[0][0][0]].append(self.improvements[0][1] -\
                        config['first_candidate_fitness'])
                    for method in strategies_data:
                        if method != self.improvements[0][0][0]:
                            strategies_data[method].append(0)
                for n, improvement in enumerate(self.improvements[1:]):
                    if improvement[1] > time_range[1]:
                        break
                    for strategy in set(improvement[0]):
                        strategies_data[strategy].append((improvement[1] - self.improvements[n + gen_range[0]][1]) *\
                            (improvement[0].count(strategy)/len(improvement[0])) + strategies_data[strategy][-1])
                    for strategy in strategies_data:
                        if not strategy in improvement[0]:
                            if len(strategies_data[strategy]) == 0:
                                strategies_data[strategy].append(previous_strategy_data[strategy])
                                continue
                            strategies_data[strategy].append(strategies_data[strategy][-1])
                if xparam == 'time':
                    xdata = [improvement[2] for improvement in improvements_data]
                elif xparam == 'generation':
                    xdata = [int(i.split('_')[1]) for i in self.improvements_files if\
                        (gen_range[0] <= int(i.split('_')[1]) <= gen_range[1])]
                elif xparam == 'n':
                    xdata = list(range(len(improvements_data)))
                else:
                    raise Exception(f'{xparam} is not valid')
            else:
                raise Exception(f'{yparam} is not valid')
            if yparam == 'best' or yparam == 'mean':
                if xparam == 'time':
                    xdata = [gen[0] for gen in self.generations if\
                        (gen_range[0] <= self.generations.index(gen) <= gen_range[1])]
                elif xparam == 'generation':
                    xdata = [n for n, gen in enumerate(self.generations) if (gen_range[0] <= n <= gen_range[1])]
                elif xparam == 'n':
                    pass
                else:
                    raise Exception(f'{xparam} is not valid')
            if 'strategies' in yparam and xparam != 'n':
                for n, strategy in enumerate(config['strategies']['strategies']):
                    if config['strategies']['rate'][n] == 0:
                        for method in config['strategies']['strategies'][strategy]['methods']:
                            if method in strategies_data:
                                strategies_data.pop(method)
                        continue
                    for n, method in enumerate(config['strategies']['strategies'][strategy]['methods']):
                        if config['strategies']['strategies'][strategy]['rate'][n] == 0:
                            if method in strategies_data:
                                strategies_data.pop(method)
                if 'fitness' in yparam:
                    plt.ylabel(strtgs_fit_str)
                if xparam == 'time':
                    plt.xlabel(rf'{time_str} ($s$)')
                elif xparam == 'generation':
                    plt.xlabel(f'{gen_str}')
                else:
                    raise Exception(f'{xparam} is not valid')
                for strategy in sorted(strategies_data.keys()):
                    ax.plot(xdata, strategies_data[strategy], drawstyle='steps',
                        label=strategy if not strategy in strategies_dict else strategies_dict[strategy])
                plt.legend()
                if format == 'tex':
                    tikzplotlib.save(f'{savedir}/{xparam}_{yparam}.tex')
                else:
                    plt.savefig(f'{savedir}/{xparam}_{yparam}.{format}', bbox_inches='tight')
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
                    new_ydata = [ydata[0]]
                    for n, item in enumerate(ydata[1:]):
                        if ydata[n] == item:
                            continue
                        new_ydata.append(item)
                    ydata = new_ydata
                    xdata = list(range(len(ydata)))
                    plt.xlabel(r'$N$')
                ax.plot(xdata, ydata, drawstyle='steps')
                if format == 'tex':
                    tikzplotlib.save(f'{savedir}{xparam}_{yparam}.tex')
                else:
                    plt.savefig(f'{savedir}{xparam}_{yparam}.{format}', bbox_inches='tight')
                print(f'{self.data_dir} data x={xparam} y={yparam} plotted with success')


def concatenate(*args: Data | str, savedir: str):
    """Concatenate different data saved by Genetic objects in a single Data object

    Parameters
    ----------
    *args : Data | str
        Data objects or directory where de data was saved. The order it is input is the order it is concatenated
    savedir : str
        Directorey where the concatenated data will be saved
    """
    args = [Data(arg) if type(arg) == str else arg for arg in args]
    files_extension = os.listdir(f'{args[0].data_dir}/improvements')[0].split('.')[1]
    shutil.copytree(args[0].data_dir, savedir)
    last_improvement = 0
    last_lineage = 0
    last_gen = 0
    last_time_gen = 0
    for n, _next in enumerate(args[1:]):
        last = args[n]
        last_improvement += len(last.improvements_files) - 1
        last_lineage += len(last.lineage_files) - 1
        last_gen += len(last.generations) - 1
        last_time_gen += float(last.generations[-1][0])
        next_improvements_files = _next.improvements_files[1:]
        next_lineage_files = _next.lineage_files[1:]
        for improvement in next_improvements_files:
            new_filename = [int(x) for x in improvement.split('_')]
            new_filename[0] += last_improvement
            new_filename[1] += last_gen
            new_filename = ('_').join([str(x) for x in new_filename])
            shutil.copy(f'{_next.data_dir}/improvements/{improvement}.{files_extension}',
                f'{savedir}/improvements/{new_filename}.{files_extension}')
        for ancestor in next_lineage_files:
            new_filename = [int(x) for x in ancestor.split('_')]
            new_filename[0] += last_lineage
            new_filename[1] += last_gen
            new_filename = ('_').join([str(x) for x in new_filename])
            shutil.copy(f'{_next.data_dir}/lineage/{ancestor}.{files_extension}',
                f'{savedir}/lineage/{new_filename}.{files_extension}')
        new_improvements = '\n'.join(['\t'.join([str(improvement[0]).replace("'", ''), str(improvement[1])] +\
            [str(float(improvement[2]) + last_time_gen)])  for improvement in _next.improvements[1:]])
        new_lineage =  '\n'.join(['\t'.join([str(ancestor[0]).replace("'", '') ,str(ancestor[1])] +\
            [str(float(ancestor[2]) + last_time_gen)]) for ancestor in _next.lineage[1:]])
        new_generations = '\n'.join(['\t'.join([str(int(gen[0]) + last_time_gen)] + [str(x) for x in gen[1:]]) for\
            gen in _next.generations])
        if not n == len(args) - 2:
            new_improvements += '\n'
            new_lineage += '\n'
            new_generations += '\n'
        with open(f'{savedir}/improvements_strategies.log', 'a') as new_file:
            new_file.write(new_improvements)
        with open(f'{savedir}/lineage_strategies.log', 'a') as new_file:
            new_file.write(new_lineage)
        with open(f'{savedir}/time_gen.log', 'a') as new_file:
            new_file.write(new_generations)
    return(Data(savedir))
