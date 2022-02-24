import datetime
import json
import multiprocessing
import operator
import os
import pickle
import random
from functools import partial

import PIL
import networkx as nx
import numpy
import pandas as pd
import seaborn as sn
from PIL import Image, ImageFilter
from deap import gp, base, tools, creator, algorithms
from deap.tools import selBest, selTournament
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from tqdm import tqdm

xPos = 0
yPos = 0


# Define new functions
def genHalfandHalfErrorFix(pset, min_, max_, type_=None):
    while True:
        try:
            return gp.genHalfAndHalf(pset, min_, max_, type_)
        except Exception:
            pass


def mutUniformFix(individual, expr, pset):
    while True:
        try:
            return gp.mutUniform(individual, expr, pset)
        except Exception:
            pass


def meanFilter(size, meanFilters):
    """
    Gets pixel of box blur filtered image at global vars xPos and yPos.

    :param size: Size of mean filter to use
    :type size: int
    :type meanFilters: list[PIL.Image.Image]
    :rtype: int
    """
    idx = size // 2
    return meanFilters[idx].getpixel((xPos, yPos))


def maxFilter(size, maxFilters):
    """
    Gets pixel of max filtered image at global vars xPos and yPos.

    :param size: Size of max filter to use
    :type size: int
    :type maxFilters: list[PIL.Image.Image]
    :rtype: int
    """
    idx = size // 2
    return maxFilters[idx].getpixel((xPos, yPos))


def minFilter(size, minFilters):
    """
    Gets pixel of min filtered image at global vars xPos and yPos.

    :param size: Size of min filter to use
    :type size: int
    :type minFilters: list[PIL.Image.Image]
    :rtype: int
    """
    idx = size // 2
    return minFilters[idx].getpixel((xPos, yPos))


def edgeFilter(edge):
    """
    Gets pixel of edge filtered image at global vars xPos and yPos.

    :type edge: PIL.Image.Image
    :rtype: int
    """
    return edge.getpixel((xPos, yPos))


def edgeFilterPlus(edgePlus):
    """
    Gets pixel of edge-plus filtered image at global vars xPos and yPos.

    :type edgePlus: PIL.Image.Image
    :rtype: int
    """
    return edgePlus.getpixel((xPos, yPos))


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


def evalClass(individual, toolbox, image, xRange, yRange):
    """
    Evaluate programs during training.

    :param individual: Individual to be evaluated
    :type toolbox: deap.base.Toolbox
    :param image: The classification image
    :type image: Image.Image
    :type xRange: tuple[int, int]
    :type yRange: tuple[int, int]
    :return:
    """
    global xPos, yPos
    hits = 0
    for x in range(xRange[0], xRange[1] + 1):
        xPos = x
        for y in range(yRange[0], yRange[1] + 1):
            yPos = y
            prediction = toolbox.compile(expr=individual) >= 0
            classification = sum(image.getpixel((xPos, yPos))) > 0  # Classification image has some sort of colour here

            if (prediction and classification) or \
                    (not prediction and not classification):
                hits += 1
    return hits,


def evalTesting(individual, toolbox, image, xRange, yRange):
    """
    Evaluate programs during testing.

    :param individual: Individual to be evaluated
    :type toolbox: deap.base.Toolbox
    :param image: The classification image
    :type image: Image.Image
    :type xRange: tuple[int, int]
    :type yRange: tuple[int, int]
    :return: (hits, predicted, actual, performance image)
    :rtype: tuple[int, list, list, Image.Image]
    """
    global xPos, yPos
    # Transform the tree expression in a callable function
    pImage = Image.new(mode="RGB", size=(xRange[1] - xRange[0] + 1, yRange[1] - yRange[0] + 1))  # The performance image
    print(pImage.size)

    predicted = []
    actual = []
    hits = 0
    prediction = toolbox.compile(expr=individual) >= 0
    for x in range(xRange[0], xRange[1] + 1):
        xPos = x
        for y in range(yRange[0], yRange[1] + 1):
            yPos = y
            classification = sum(image.getpixel((xPos, yPos))) > 0  # Classification image has some sort of colour here

            if (prediction and classification) or \
                    (not prediction and not classification):
                hits += 1

            # Draw the performance image
            if prediction and classification:  # True positive
                pImage.putpixel((x - xRange[0], y - yRange[0]), (0, 255, 0))
            elif not prediction and not classification:  # True negative
                pImage.putpixel((x - xRange[0], y - yRange[0]), (0, 0, 0))
            elif not prediction and classification:  # False Negative
                pImage.putpixel((x - xRange[0], y - yRange[0]), (255, 255, 0))
            elif prediction and not classification:  # False Positive
                pImage.putpixel((x - xRange[0], y - yRange[0]), (255, 0, 0))

            actual.append(classification)
            predicted.append(prediction)
    return hits, predicted, actual, pImage


def createPrimitiveSet(toolbox):
    """
    Creates the primitive set for GP.

    :type toolbox: deap.base.Toolbox
    :rtype: deap.gp.PrimitiveSetTyped
    """
    pset = gp.PrimitiveSetTyped("MAIN", [], float, 'x')
    pset.addPrimitive(operator.add, [float, float], float, "ADD")
    pset.addPrimitive(operator.sub, [float, float], float, 'SUB')
    pset.addPrimitive(operator.mul, [float, float], float, 'MUL')
    pset.addPrimitive(protectedDiv, [float, float], float, 'DIV')
    pset.addPrimitive(operator.neg, [float], float, 'NEG')
    pset.addPrimitive(max, [float, float], float, 'MAX')
    pset.addPrimitive(min, [float, float], float, 'MIN')
    pset.addEphemeralConstant("rand", lambda: random.uniform(-10, 10), float)
    pset.addPrimitive(toolbox.meanFilter, [int], float, "MeanF")
    pset.addPrimitive(toolbox.maxFilter, [int], float, "MaxF")
    pset.addPrimitive(toolbox.minFilter, [int], float, "MinF")
    pset.addPrimitive(toolbox.edgeFilter, [], float, "EdgeF")
    pset.addPrimitive(toolbox.edgeFilterPlus, [], float, "EdgePF")
    pset.addTerminal(1, int)
    pset.addTerminal(3, int)
    pset.addTerminal(5, int)
    pset.addTerminal(7, int)
    pset.addTerminal(9, int)
    pset.addTerminal(11, int)

    return pset


def createToolbox(min_init_size, max_init_size, image, classImage, crossoverMethod, min_mut_size,
                  max_mut_size, numOfElites, tournSize, xPosTraining, yPosTraining,
                  xPosTesting, yPosTesting):
    """
    Creates the toolbox for GP. Mainly, this consists of initializing the population.

    :type image: Image.Image
    :param image: The default image.
    :type classImage: Image.Image
    :param classImage: The classification image.
    :type min_init_size: int
    :type max_init_size: int
    :type min_mut_size: int
    :type max_mut_size: int
    :type numOfElites: int
    :type tournSize: int
    :type xPosTraining: tuple[int, int]
    :type yPosTraining: tuple[int, int]
    :type xPosTesting: tuple[int, int]
    :type yPosTesting: tuple[int, int]
    :return: The toolbox for GP.
    :rtype: deap.base.Toolbox
    """
    assert image.size != (0, 0)
    toolbox = base.Toolbox()

    # Collect all of the filter images first
    maxFilters = []
    minFilters = []
    meanFilters = []

    for i in range(1, 12, 2):
        maxFilters.append(image.filter(ImageFilter.MaxFilter(i)))
        minFilters.append(image.filter(ImageFilter.MinFilter(i)))
        meanFilters.append(image.filter(ImageFilter.BoxBlur(i)))

    edge = image.filter(ImageFilter.EDGE_ENHANCE)
    edgePlus = image.filter(ImageFilter.EDGE_ENHANCE_MORE)

    # Register filter functions with toolbox
    toolbox.register("meanFilter", meanFilter, meanFilters=meanFilters)
    toolbox.register("maxFilter", maxFilter, maxFilters=maxFilters)
    toolbox.register("minFilter", minFilter, minFilters=minFilters)
    toolbox.register("edgeFilter", edgeFilter, edge=edge)
    toolbox.register("edgeFilterPlus", edgeFilterPlus, edgePlus=edgePlus)

    # Register necessary toolbox elements for GP
    pset = createPrimitiveSet(toolbox)
    toolbox.register("expr", genHalfandHalfErrorFix, pset=pset, min_=min_init_size, max_=max_init_size)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("evaluate", evalClass, toolbox=toolbox, image=classImage, xRange=xPosTraining, yRange=yPosTraining)
    toolbox.register("evaluateTesting", evalTesting, toolbox=toolbox, image=classImage,
                     xRange=xPosTesting, yRange=yPosTesting)
    toolbox.register("mate", crossoverMethod)
    toolbox.register("expr_mut", gp.genFull, min_=min_mut_size, max_=max_mut_size)
    toolbox.register("mutate", mutUniformFix, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.register("select", selTournamentElitism, numOfElites=numOfElites, tournsize=tournSize)

    return toolbox


def run(params):
    im: Image.Image = Image.open(params['standardImageFilePath'])

    if params['grayscale']:
        im = im.convert(mode="L")  # Standard image in grayscale

    classIm = Image.open(params['classImageFilePath'])

    # Initializing things for GP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = createToolbox(params['minInitSize'], params['maxInitSize'], im, classIm, gp.cxOnePoint,
                            params['minMutSize'],
                            params['maxMutSize'], params['numOfElites'], params['tournSize'], params['xPosTraining'],
                            params['yPosTraining'], params['xPosTesting'], params['yPosTesting'])

    # Keep track of best solution and logs of all GP executions
    bestHof = (None, -1, None, None, None)
    # (genetic program, hits, list[predicted on testing], list[actual testing result], performance image)
    logs = []

    # Complete a set number of GP runs
    for _ in tqdm(range(params['numOfRuns'])):
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

        eval = toolbox.evaluateTesting(hof[0])
        if eval[0] > bestHof[1]:
            bestHof = (hof[0], eval[0], eval[1], eval[2], eval[3])
        logs.append(log)

    # Save all logs in pickled form
    fileSuffix = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    os.mkdir("./" + fileSuffix)
    with open("./" + fileSuffix + '/logs.pkl', 'wb') as f:
        pickle.dump(logs, f)

    # Save the performance image
    bestHof[4].save(fp="./" + fileSuffix + '/performance.png')

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
    plt.savefig("./" + fileSuffix + '/HOFgraph.png')

    # Creating the confusion matrix and showing heatmap of best program in all runs
    predicted = bestHof[2]
    actual = bestHof[3]
    matrix_data = {'y_Actual': actual,
                   'y_Predicted': predicted}
    df = pd.DataFrame(matrix_data, columns=['y_Actual', 'y_Predicted'])
    print(df)
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix)
    sn.heatmap(confusion_matrix, annot=True, cmap='Blues')
    plt.savefig("./" + fileSuffix + '/HOFTestingConfusion.png')

    # Save additional information to text file (identifies run configuration)
    with open("./" + fileSuffix + '/info.txt', 'w') as f:
        f.write(str(params) + '\n')
        f.write(str(bestHof[0]) + '\n')
        f.write(str(bestHof[1]) + '/' + str(
            params['xPosTesting'][1] - params['xPosTesting'][0] * params['yPosTesting'][1] - params['yPosTesting'][0]))


if __name__ == "__main__":
    # When executing this as main, it runs and reads from parameter file.
    params = {
        'popSize': 300,
        'crossoverP': 0.9,
        'mutationP': 0.1,
        'numOfGenerations': 2,
        'numOfRuns': 30,
        'numOfElites': 3,
        'minInitSize': 2,
        'maxInitSize': 4,
        'minMutSize': 1,
        'maxMutSize': 2,
        'tournSize': 3,
        'standardImageFilePath': "images/standard.png",
        'classImageFilePath': "images/classification.png",
        'grayscale': True,
        'xPosTraining': (28, 276),  # (from, to)
        'yPosTraining': (55, 233),  # (from, to)
        'xPosTesting': (996, 1211),  # (from, to)
        'yPosTesting': (210, 442),  # (from, to)
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
