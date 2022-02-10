import datetime
import itertools
import json
import math
import multiprocessing
import operator
import pickle
import random
import pandas as pd
import seaborn as sn
from scoop import futures

import networkx as nx
import numpy
from deap import gp, creator, base, tools, algorithms
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
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


def evalTesting(individual, toolbox, testingSet):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    predicted = []
    hits = 0
    for x in testingSet:
        val = func(*x[1:])
        if val >= 0:
            predicted.append('M')
            if x[0] == 'M':
                hits += 1
        else:
            predicted.append('B')
            if x[0] == 'B':
                hits += 1
    return predicted, hits


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

    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, numOfArgs), float, 'x')
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
    pset.addEphemeralConstant("rand", lambda: float(round(random.uniform(-5, 5), 2)), float)

    return pset


def createToolbox(pset, min_init_size, max_init_size, trainingSet, testingSet, crossoverMethod, min_mut_size,
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
    toolbox.register("evalTesting", evalTesting, toolbox=toolbox, testingSet=testingSet)
    toolbox.register("mate", crossoverMethod)
    toolbox.register("expr_mut", gp.genFull, min_=min_mut_size, max_=max_mut_size)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.register("select", selTournamentElitism, numOfElites=numOfElites, tournsize=tournSize)

    return toolbox


def run(params):
    # Used for splitting the data into training/testing sets
    split_percent = params['trainSetPercentage']

    # First read the data, and then shuffle it
    classification, attributes = read(params['dataReadOption'])

    # Data is separated into training and testing, while being
    # stratified to keep relatively similar distributions of classifications
    X_train, X_test, y_train, y_test = train_test_split(attributes, classification, train_size=split_percent,
                                                        shuffle=True, stratify=classification)
    trainingSet = [list(y_train[i]) + X_train[i] for i in range(len(y_train))]
    testingSet = [list(y_test[i]) + X_test[i] for i in range(len(y_test))]

    # Creating the initial primitive set
    pset = createPrimitiveSet(len(attributes[0]))

    # Initializing things for GP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    # Create the toolbox
    toolbox = createToolbox(pset, params['minInitSize'], params['maxInitSize'], trainingSet, testingSet,
                            gp.cxOnePoint, params['minMutSize'], params['maxMutSize'], params['numOfElites'],
                            params['tournSize'])

    pop = toolbox.population(n=params['popSize'])
    hof = tools.HallOfFame(1)

    # This is to keep track of best program, and all the logs
    # of all runs, respectively.
    bestHof = (None, -1, 0)
    logs = []

    # Complete a set number of GP runs
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
                                     halloffame=hof, verbose=False)

        eval = toolbox.evalTesting(hof[0])
        if eval[1] > bestHof[1]:
            bestHof = (hof[0], eval[1], eval[0])
        logs.append(log)

    fileSuffix = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    with open(fileSuffix + '.pkl', 'wb') as f:
        pickle.dump(logs, f)

    # Creating the node graph of the best overall program throughout all runs
    nodes, edges, labels = gp.graph(bestHof[0])

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.nx_pydot.graphviz_layout(g)
    nx.draw_networkx_nodes(g, pos, node_size=25,
                           node_color=['#FF0000FF'] + ['#FFFFFF00' for x in range(len(nodes) - 1)])
    nx.draw_networkx_edges(g, pos, edge_color='#aaaaaa')
    nx.draw_networkx_labels(g, pos, labels, font_color='#000000', font_size=8)
    plt.savefig(fileSuffix + '-HOFgraph.png')
    plt.show()

    # Creating the confusion matrix and showing heatmap of best program in all runs
    predicted = bestHof[2]
    actual = [x[0] for x in testingSet]
    matrix_data = {'y_Actual': actual,
                   'y_Predicted': predicted}
    df = pd.DataFrame(matrix_data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True)
    plt.savefig(fileSuffix + '-HOFTestingConfusion.png')
    plt.show()

    with open(fileSuffix + '.txt', 'w') as f:
        f.write(str(params))
        f.write('\n' + str(bestHof[0]))
        f.write('\n' + str(bestHof[1]) + '/' + str(len(testingSet)))


if __name__ == "__main__":
    # When executing this as main, it runs and reads from parameter file.
    params = {
        'popSize': 300,
        'crossoverP': 0.9,
        'mutationP': 0.1,
        'numOfGenerations': 1000,
        'numOfRuns': 30,
        'numOfElites': 3,
        'trainSetPercentage': 0.8,
        'minInitSize': 2,
        'maxInitSize': 4,
        'minMutSize': 1,
        'maxMutSize': 2,
        'tournSize': 3,
        'dataReadOption': 'all',
    }

    try:
        f = open('params.json', 'r')
        params = json.load(f)
        f.close()
    except FileNotFoundError:
        f = open('params.json', 'w')
        json.dump(params, f)
        f.close()

    run(params)
