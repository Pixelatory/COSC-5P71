import datetime
import json
import operator
import os
import pickle
import random

import networkx as nx
import numpy
import pandas as pd
import seaborn as sn
from PIL import Image, ImageFilter
from deap import gp, base, tools, creator, algorithms
from matplotlib import pyplot as plt
from tqdm import tqdm

from A2.util import protectedDiv, meanFilter, maxFilter, minFilter, edgeFilter, edgeFilterPlus, embossFilter, \
    edgesFilter, genHalfandHalfErrorFix, evalClass, evalTesting, mutUniformFix, selTournamentElitism, FilterSizes, \
    FilterSizeOne, FilterSizeThree, FilterSizeFive, FilterSizeSeven, FilterSizeNine, FilterSizeEleven, Tuple, \
    redChannel, greenChannel, blueChannel, intensity


# Many functions were put into their own python file (util.py)
def createPrimitiveSet(toolbox, grayscale):
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
    pset.addPrimitive(operator.neg, [int], int, 'NEG')
    pset.addEphemeralConstant("randf", lambda: random.uniform(-10, 10), float)
    pset.addEphemeralConstant("randi", lambda: random.randint(-10, 10), int)
    pset.addTerminal(FilterSizeOne, FilterSizes)
    pset.addTerminal(FilterSizeThree, FilterSizes)
    pset.addTerminal(FilterSizeFive, FilterSizes)
    pset.addTerminal(FilterSizeSeven, FilterSizes)
    pset.addTerminal(FilterSizeNine, FilterSizes)
    pset.addTerminal(FilterSizeEleven, FilterSizes)

    if grayscale:
        pset.addPrimitive(toolbox.meanFilter, [FilterSizes, int, int], float, "MeanF")
        pset.addPrimitive(toolbox.maxFilter, [FilterSizes, int, int], float, "MaxF")
        pset.addPrimitive(toolbox.minFilter, [FilterSizes, int, int], float, "MinF")
        pset.addPrimitive(toolbox.edgeFilter, [int, int], float, "EdgeF")
        pset.addPrimitive(toolbox.edgeFilterPlus, [int, int], float, "EdgePF")
        pset.addPrimitive(toolbox.embossFilter, [int, int], float, "EmF")
        pset.addPrimitive(toolbox.edgesFilter, [int, int], float, "EdgesF")

    if not grayscale:
        pset.addPrimitive(toolbox.meanFilter, [FilterSizes, int, int], Tuple, "MeanF")
        pset.addPrimitive(toolbox.maxFilter, [FilterSizes, int, int], Tuple, "MaxF")
        pset.addPrimitive(toolbox.minFilter, [FilterSizes, int, int], Tuple, "MinF")
        pset.addPrimitive(toolbox.edgeFilter, [int, int], Tuple, "EdgeF")
        pset.addPrimitive(toolbox.edgeFilterPlus, [int, int], Tuple, "EdgePF")
        pset.addPrimitive(toolbox.embossFilter, [int, int], Tuple, "EmF")
        pset.addPrimitive(toolbox.edgesFilter, [int, int], Tuple, "EdgesF")
        pset.addPrimitive(redChannel, [Tuple], float, "Red")
        pset.addPrimitive(greenChannel, [Tuple], float, "Green")
        pset.addPrimitive(blueChannel, [Tuple], float, "Blue")
        pset.addPrimitive(intensity, [Tuple], float, "avgRGB")



    return pset


def createToolbox(grayscale, min_init_size, max_init_size, image, classImage, edgesImage, crossoverMethod, min_mut_size,
                  max_mut_size, numOfElites, tournSize, xPosTraining, yPosTraining,
                  xPosTesting, yPosTesting, numOfSamples):
    """
    Creates the toolbox for GP. Mainly, this consists of initializing the population.

    :type grayscale: bool
    :param grayscale: If the images used are in grayscale.
    :type image: Image.Image
    :param image: The default image.
    :type classImage: Image.Image
    :param classImage: The classification image.
    :param edgesImage: The edges image.
    :type edgesImage: Image.Image
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
    :type numOfSamples: tuple[int, int]
    :param numOfSamples: Number of samples to take of (true class, false class)
    :return: The toolbox for GP.
    :rtype: deap.base.Toolbox
    """
    assert image.size != (0, 0)
    toolbox = base.Toolbox()

    print("Preprocessing...")
    # Collect all positions of true and false classifications
    classPos = []  # Positions of true classifications
    nonClassPos = []  # Positions of false classifications

    for x in range(xPosTraining[0], xPosTraining[1] + 1):
        for y in range(yPosTraining[0], yPosTraining[1] + 1):
            if sum(classImage.getpixel((x, y))) > 0:
                classPos.append((x, y))
            else:
                nonClassPos.append((x, y))
    print("Collected classification and non-classification positions for sampling.")

    # Collect all of the filter images
    maxFilters = []
    minFilters = []
    meanFilters = []

    for i in range(1, 12, 2):
        maxFilters.append(image.filter(ImageFilter.MaxFilter(i)))
        minFilters.append(image.filter(ImageFilter.MinFilter(i)))
        meanFilters.append(image.filter(ImageFilter.BoxBlur(i)))

    edge = image.filter(ImageFilter.EDGE_ENHANCE)
    edgePlus = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    emboss = image.filter(ImageFilter.EMBOSS)

    print("Created all filters for images.")
    print("Image preprocessing finished.")

    # Register filter functions with toolbox
    toolbox.register("meanFilter", meanFilter, meanFilters=meanFilters)
    toolbox.register("maxFilter", maxFilter, maxFilters=maxFilters)
    toolbox.register("minFilter", minFilter, minFilters=minFilters)
    toolbox.register("edgeFilter", edgeFilter, edge=edge)
    toolbox.register("edgeFilterPlus", edgeFilterPlus, edgePlus=edgePlus)
    toolbox.register("embossFilter", embossFilter, emboss=emboss)
    toolbox.register("edgesFilter", edgesFilter, edges=edgesImage)

    # Create the primitive set
    pset = createPrimitiveSet(toolbox, grayscale)

    # Register necessary toolbox elements for GP
    toolbox.register("expr", genHalfandHalfErrorFix, pset=pset, min_=min_init_size, max_=max_init_size)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("evaluate", evalClass, toolbox=toolbox, classPos=classPos, nonClassPos=nonClassPos,
                     numOfSamples=numOfSamples)
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
    # Open the standard image
    im: Image.Image = Image.open(params['standardImageFilePath']).convert(mode="RGB")

    if params['grayscale']:
        im = im.convert(mode="L")  # Standard image in grayscale

    # Open the classification image
    classIm = Image.open(params['classImageFilePath']).convert(mode="RGB")

    # Open the edges image
    edgesIm = Image.open(params['edgeImageFilePath']).convert(mode="RGB")

    if params['grayscale']:
        edgesIm = edgesIm.convert(mode="L")

    print("Opened standard, classification, and edge images.")

    # Initializing things for GP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = createToolbox(params['grayscale'], params['minInitSize'], params['maxInitSize'], im, classIm, edgesIm,
                            gp.cxOnePoint, params['minMutSize'], params['maxMutSize'], params['numOfElites'],
                            params['tournSize'], params['xPosTraining'], params['yPosTraining'], params['xPosTesting'],
                            params['yPosTesting'], params['numOfSamples'])

    # Keep track of best solution and logs of all GP executions
    bestHof = (None, -1, None, None, None)
    # bestHof: (genetic program, hits, list[predicted on testing], list[actual testing result], performance image)

    logs = []

    print("Performing GP executions.")

    # Complete a set number of GP runs
    for i in tqdm(range(params['numOfRuns'])):
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

        print("Run " + str(i) + " complete.")

    # Save all logs in pickled form
    fileSuffix = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    os.mkdir("./" + fileSuffix)

    print("Now collecting data to save in /" + fileSuffix)

    with open("./" + fileSuffix + '/logs.pkl', 'wb') as f:
        pickle.dump(logs, f)

    # Save the performance image
    bestHof[4].save(fp="./" + fileSuffix + '/performance.png')

    # Save additional information to text file (identifies run configuration)
    with open("./" + fileSuffix + '/info.txt', 'w') as f:
        f.write(str(params) + '\n')
        f.write(str(bestHof[0]) + '\n')
        f.write(str(bestHof[1]) + '/' + str(
            params['xPosTesting'][1] - params['xPosTesting'][0] * params['yPosTesting'][1] - params['yPosTesting'][0]))

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

    plt.close('all')

    # Creating the confusion matrix and showing heatmap of best program in all runs
    predicted = bestHof[2]
    actual = bestHof[3]
    matrix_data = {'y_Actual': actual,
                   'y_Predicted': predicted}
    df = pd.DataFrame(matrix_data, columns=['y_Actual', 'y_Predicted'])
    print(df)
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    with open("./" + fileSuffix + '/matrix.pkl', 'wb') as f:
        pickle.dump(confusion_matrix, f)
    print(confusion_matrix)
    sn.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.savefig("./" + fileSuffix + '/HOFTestingConfusion.png')


if __name__ == "__main__":
    # When executing this as main, it runs and reads from parameter file.
    params = {
        'popSize': 300,
        'crossoverP': 0.9,
        'mutationP': 0.1,
        'numOfGenerations': 100,
        'numOfRuns': 15,
        'numOfElites': 3,
        'minInitSize': 2,
        'maxInitSize': 4,
        'minMutSize': 1,
        'maxMutSize': 2,
        'tournSize': 3,
        'standardImageFilePath': "images/standard.png",
        'classImageFilePath': "images/classification.png",
        "edgeImageFilePath": "images/edges.png",
        'grayscale': True,
        'xPosTraining': (28, 276),  # (from, to)
        'yPosTraining': (55, 233),  # (from, to)
        'xPosTesting': (996, 1211),  # (from, to)
        'yPosTesting': (210, 442),  # (from, to)
        'numOfSamples': (100, 100),  # (true classification, false classification)
    }

    try:
        f = open('params-2.json', 'r')
        params = json.load(f)
        f.close()
    except FileNotFoundError:
        f = open('params-2.json', 'w')
        json.dump(params, f)
        f.close()

    run(params)
