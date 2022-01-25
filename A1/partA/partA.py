import operator
import math
import random
import json
import time
import numpy
import matplotlib.pyplot as plt
import networkx as nx
import pickle

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap.tools import selBest, selTournament


# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1.0


def selTournamentElitism(individuals, k, tournsize, numOfElites, fit_attr="fitness"):
    """
    Combination of tournament selection with elitism, since one isn't provided by DEAP.

    :param individuals: Individuals that may be in tournament.
    :param k: Number of individuals to select.
    :param tournsize: Amount of individuals to be randomly selected for tournament.
    :param numOfElites: Number of elites automatically added to new generation.
    :param fit_attr: Attribute of individuals to use as selection criterion.
    :return:
    """
    chosen = selBest(individuals, numOfElites, fit_attr)
    chosen += selTournament(individuals, k - numOfElites, tournsize, fit_attr)
    return chosen


def evalSymbReg(individual, points, toolbox):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    sqerrors = ((func(x) - x ** 4 - x ** 3 - x ** 2 - x) ** 2 for x in points)
    print(type(math.fsum(sqerrors) / len(points),))
    return math.fsum(sqerrors) / len(points),


def main():
    pset = gp.PrimitiveSetTyped("MAIN", [float], float)
    pset.addPrimitive(operator.add, [float, float], float, name="add")
    pset.addPrimitive(operator.sub, [float, float], float, name='sub')
    pset.addPrimitive(operator.mul, [float, float], float, name='mul')
    pset.addPrimitive(protectedDiv, [float, float], float, name='div')
    pset.addPrimitive(operator.neg, [float], float, name='neg')
    pset.addPrimitive(math.cos, [float], float, name='cos')
    pset.addPrimitive(math.sin, [float], float, name='sin')
    pset.addEphemeralConstant("rand101", lambda: float(random.randint(-1, 1)), float)
    pset.renameArguments(ARG0='x')

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("evaluate", evalSymbReg, toolbox=toolbox, points=[x / 10. for x in range(-10, 10)])
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    params = {
        'popSize': 300,
        'crossoverP': 0.5,
        'mutationP': 0.1,
        'numOfGenerations': 40,
        'numOfRuns': 10,
        'numOfElites': 1,
    }
    try:
        f = open('params.json', 'r')
        params = json.load(f)
        f.close()
    except FileNotFoundError:
        f = open('params.json', 'w')
        json.dump(params, f)
        f.close()

    toolbox.register("select", selTournamentElitism, numOfElites=params['numOfElites'], tournsize=3)

    seeds = []
    bestHof = (None, 1000000)
    logs = []

    for i in range(params['numOfRuns']):
        seed = int(time.time())
        seeds.append(seed)
        random.seed(seed)

        pop = toolbox.population(n=params['popSize'])
        hof = tools.HallOfFame(1)

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

        if toolbox.evaluate(hof[0])[0] < bestHof[1]:
            bestHof = (hof[0], toolbox.evaluate(hof[0])[0])
        logs.append(log)

    return logs, bestHof


if __name__ == "__main__":
    fileSuffix = '-100'
    logs, bestHof = main()
    with open('logs' + fileSuffix + '.pkl', 'wb') as f:
        pickle.dump(logs, f)

    '''
    plt.close('all')

    plt.plot(log.select("gen"), log.chapters["fitness"].select("avg"), 'o-')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.show()
    '''

    nodes, edges, labels = gp.graph(bestHof[0])
    print(bestHof[0])

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.nx_pydot.graphviz_layout(g)
    nx.draw_networkx_nodes(g, pos, node_size=700, node_color=['#59360D'] + ['#FFA737' for x in range(len(nodes) - 1)])
    nx.draw_networkx_edges(g, pos, edge_color='#59360D')
    nx.draw_networkx_labels(g, pos, labels, font_color='#FCFAF9')
    plt.savefig(str(bestHof[1]) + fileSuffix + '.png')
    plt.show()

    # nx.Graph()
