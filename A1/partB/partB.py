import inspect
import itertools
import json
import math
import operator
import pickle
import random

import networkx as nx
import numpy
from deap import gp, creator, base, tools, algorithms
from matplotlib import pyplot as plt

from A1.partA.partA import protectedDiv, selTournamentElitism
from readData import read


# Define new functions
def genHalfandHalfErrorFix(pset, min_, max_, type_=None):
    while True:
        try:
            return gp.genHalfAndHalf(pset, min_, max_, type_)
        except Exception:
            pass


def evalClass(individual, toolbox, trainingSet):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    hits = 0
    for x in trainingSet:
        val = func(*x[1:])
        if (val >= 0 and x[0] == 'M') \
                or (val < 0 and x[0] == 'B'):
            hits += 1
    return hits,


def splitPercentage(data, split_percent):
    """
    Splits an iterable object into two portions, based
    on a percentage given as argument.

    :param data: Data to be split.
    :param split_percent: Percentage of the data split.
    :type split_percent: int
    :return: Two split pieces of data.
    """
    prop = int((split_percent / 100) * len(data))
    return data[prop:], data[:prop]


def countClassification(data, classification):
    count = 0
    for entry in data:
        if entry[0] == classification:
            count += 1
    return count


def if_then_else(one, two, three):
    """
    An if, then, and else function defined for the function set
    of the genetic program.

    :param one: The if condition.
    :type one: bool
    :param two: The return of if condition.
    :type two: float
    :param three: The return of else.
    :type three: float
    :return: float
    """
    if one:
        return two
    else:
        return three


def createPrimitiveSet(numOfArgs):
    """

    :return:
    """
    # Just ensure we don't give an invalid number of args that are supported
    possibleArgs = [30, 10, 20]
    assert numOfArgs in possibleArgs

    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, numOfArgs), float)
    pset.addPrimitive(operator.add, [float, float], float, "ADD")
    pset.addPrimitive(operator.sub, [float, float], float, 'SUB')
    pset.addPrimitive(operator.mul, [float, float], float, 'MUL')
    pset.addPrimitive(protectedDiv, [float, float], float, 'DIV')
    pset.addPrimitive(operator.neg, [float], float, 'NEG')
    pset.addPrimitive(math.cos, [float], float, 'COS')
    pset.addPrimitive(math.sin, [float], float, 'SIN')
    pset.addPrimitive(max, [float, float], float, 'MAX')
    pset.addPrimitive(min, [float, float], float, 'MIN')
    pset.addPrimitive(if_then_else, [bool, float, float], float, 'ITE')
    pset.addPrimitive(operator.eq, [float, float], bool, 'EQ')
    pset.addPrimitive(operator.gt, [float, float], bool, 'GT')
    pset.addPrimitive(operator.ge, [float, float], bool, 'GTE')
    pset.addPrimitive(operator.le, [float, float], bool, 'LTE')
    pset.addPrimitive(operator.lt, [float, float], bool, 'LT')
    pset.addPrimitive(operator.and_, [bool, bool], bool, 'AND')
    pset.addPrimitive(operator.or_, [bool, bool], bool, 'OR')
    pset.addPrimitive(operator.not_, [bool], bool, 'NOT')
    pset.addTerminal(False, bool, 'F')
    pset.addTerminal(True, bool, 'T')
    pset.addEphemeralConstant("rand", lambda: float(random.uniform(-5, 5)), float)

    return pset


def createToolbox(pset, min_init_size, max_init_size, trainingSet, crossoverMethod, min_mut_size,
                  max_mut_size, numOfElites, tournSize):
    """
    Creates the toolbox for GP. Mainly, this consists of initializing the population.

    :param pset: Primitive set.
    :type pset: deap.gp.PrimitiveSetTyped or deap.gp.PrimitiveSet
    :param min_init_size: Minimum initialization size for tree.
    :type min_init_size: int
    :param max_init_size: Maximum initialization size for tree.
    :type max_init_size: int
    :return: The toolbox for GP.
    :rtype: deap.base.Toolbox
    """
    toolbox = base.Toolbox()
    toolbox.register("expr", genHalfandHalfErrorFix, pset=pset, min_=min_init_size, max_=max_init_size)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("evaluate", evalClass, toolbox=toolbox, trainingSet=trainingSet)
    toolbox.register("mate", crossoverMethod)
    toolbox.register("expr_mut", gp.genFull, min_=min_mut_size, max_=max_mut_size)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.register("select", selTournamentElitism, numOfElites=numOfElites, tournsize=tournSize)

    return toolbox


if __name__ == "__main__":
    params = {
        'popSize': 300,
        'crossoverP': 0.9,
        'mutationP': 0.1,
        'numOfGenerations': 100,
        'numOfRuns': 10,
        'numOfElites': 0,
        'dataSplitPercentage': 80,
        'minInitSize': 2,
        'maxInitSize': 4,
        'minMutSize': 1,
        'maxMutSize': 2,
        'tournSize': 3,
    }

    try:
        f = open('params.json', 'r')
        params = json.load(f)
        f.close()
    except FileNotFoundError:
        f = open('params.json', 'w')
        json.dump(params, f)
        f.close()

    split_percent = params['dataSplitPercentage']  # Used for splitting the data into training/testing

    # First read the data, and then shuffle it
    data = read('all')
    random.shuffle(data)

    count = countClassification(data, 'M')
    print('M:', round(count / len(data), 1), 'B:', round((len(data) - count) / len(data), 1))

    ''' 
        Ensure that in testing set, there's relatively equal amount of both
        classifications for fairness.
    '''
    while True:
        testingSet, trainingSet = splitPercentage(data, split_percent)
        count = countClassification(testingSet, 'M')
        if 0.48 <= count / len(testingSet) <= 0.5:
            break
        random.shuffle(data)

    pset = createPrimitiveSet(len(data[0]) - 1)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = createToolbox(pset, params['minInitSize'], params['maxInitSize'], trainingSet,
                            gp.cxOnePoint, params['minMutSize'], params['maxMutSize'], params['numOfElites'],
                            params['tournSize'])

    pop = toolbox.population(n=params['popSize'])
    hof = tools.HallOfFame(1)

    bestHof = (None, 1000000)
    logs = []

    for i in range(params['numOfRuns']):
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", numpy.mean)
        mstats.register("std", numpy.std)
        mstats.register("min", numpy.min)
        mstats.register("max", numpy.max)

        _, log = algorithms.eaSimple(pop, toolbox, params['crossoverP'], params['mutationP'],
                                     params['numOfGenerations'], stats=mstats,
                                     halloffame=hof, verbose=True)

        if toolbox.evaluate(hof[0])[0] < bestHof[1]:
            bestHof = (hof[0], toolbox.evaluate(hof[0])[0])
        logs.append(log)

    fileSuffix = '-1'
    with open('logs' + fileSuffix + '.pkl', 'wb') as f:
        pickle.dump(logs, f)

    nodes, edges, labels = gp.graph(bestHof[0])

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.nx_pydot.graphviz_layout(g)
    nx.draw_networkx_nodes(g, pos, node_size=700, node_color=['#59360D'] + ['#FFA737' for x in range(len(nodes) - 1)])
    nx.draw_networkx_edges(g, pos, edge_color='#59360D')
    nx.draw_networkx_labels(g, pos, labels, font_color='#FCFAF9')
    plt.savefig(str(bestHof[1]) + fileSuffix + '.png')
    plt.show()
