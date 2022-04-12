import datetime
import json
import operator
import os
import pickle
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy
import pandas as pd
import seaborn as sn
from PIL import ImageFilter, Image
from deap import base
from deap import creator
from deap import gp
from deap import tools
from deap.algorithms import eaSimple
from tqdm import tqdm
from Project import GPalgorithm
from Project.util import protectedDiv, FilterSizeOne, FilterSizes, FilterSizeThree, FilterSizeFive, FilterSizeSeven, \
    FilterSizeNine, FilterSizeEleven, Tuple, redChannel, greenChannel, blueChannel, intensity, meanFilter, maxFilter, \
    minFilter, edgeFilter, edgeFilterPlus, embossFilter, edgesFilter, genHalfandHalfErrorFix, evalClass, evalTesting, \
    mutUniformFix, selTournamentElitism, evalTestingTwoClass


# Many functions were put into their own python file (util.py)
def save_confusion_matrix(predicted, actual, dir_name, file_suffix):
    matrix_data = {'y_Actual': actual,
                   'y_Predicted': predicted}
    df = pd.DataFrame(matrix_data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    with open("./" + dir_name + '/matrix' + file_suffix + '.pkl', 'wb') as f:
        pickle.dump(confusion_matrix, f)
    sn.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.savefig("./" + dir_name + '/HOFTestingConfusion' + file_suffix + '.png')

def save_information(hofs, logs, dir_name, file_suffix):
    """
    A simple way to save information gathered from the GP executions.

    :return: the best hof
    """
    with open("./" + dir_name + '/logs' + file_suffix + '.pkl', 'wb') as f:
        pickle.dump(logs, f)

    with open("./" + dir_name + '/hofs' + file_suffix + '.pkl', 'wb') as f:
        tmp = [(str(entry[0]), entry[1], entry[2], entry[3], entry[4]) for entry in hofs]
        pickle.dump(tmp, f)

    best_hof = (None, -1, None, None, None)
    for hof in hofs:
        if hof[1] > best_hof[1]:
            best_hof = hof

    # Save the performance image
    best_hof[4].save(fp="./" + dir_name + '/performance' + file_suffix + '.png')

    # Save additional information to text file (identifies run configuration)
    with open("./" + dir_name + '/info' + file_suffix + '.txt', 'w') as f:
        f.write(str(params) + '\n')
        f.write(str(best_hof[0]))

    # Creating the node graph of the best overall program throughout all runs
    nodes, edges, labels = gp.graph(best_hof[0])

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.nx_pydot.graphviz_layout(g)
    nx.draw_networkx_nodes(g, pos, node_size=25,
                           node_color=['#FF0000FF'] + ['#FFFFFF00' for _ in range(len(nodes) - 1)])
    nx.draw_networkx_edges(g, pos, edge_color='#aaaaaa')
    nx.draw_networkx_labels(g, pos, labels, font_color='#000000', font_size=8)
    plt.savefig("./" + dir_name + '/HOFgraph' + file_suffix + '.png')

    plt.close('all')

    # Creating the confusion matrix and showing heatmap of best program in all runs
    save_confusion_matrix(best_hof[2], best_hof[3], dir_name, file_suffix)

    return best_hof


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


def createToolbox(grayscale, min_init_size, max_init_size, image, class_boat_image, class_dock_image, edges_image,
                  crossover_method, min_mut_size, max_mut_size, num_of_elites, tourn_size, training_regions,
                  testing_region, num_of_samples, dock_separate):
    """
    Creates the toolbox for GP. Mainly, this consists of initializing the population.

    :type grayscale: bool
    :param grayscale: If the images used are in grayscale.
    :type image: Image.Image
    :param image: The default image.
    :type class_boat_image: Image.Image
    :param class_boat_image: The classification image of boats.
    :type class_dock_image: Image.Image
    :param class_dock_image: The classification image of the dock.
    :param edges_image: The edges image.
    :type edges_image: Image.Image
    :type min_init_size: int
    :type max_init_size: int
    :type min_mut_size: int
    :type max_mut_size: int
    :type num_of_elites: int
    :type tourn_size: int
    :type training_regions: list[list[int, int, int, int]]
    :type testing_region: list[int, int, int, int]
    :type num_of_samples: tuple[int, int] or tuple[int, int, int]
    :param num_of_samples: Number of samples to take of (boat, else) or (boat, dock, else)
    :param dock_separate: If the dock is classified separately to boats
    :type dock_separate: bool
    :return: The toolbox for GP.
    :rtype: list[deap.base.Toolbox]
    """
    assert image.size != (0, 0)
    assert len(num_of_samples) == 2 or len(num_of_samples) == 3

    if dock_separate:
        toolboxes = [base.Toolbox(), base.Toolbox()]
    else:
        toolboxes = [base.Toolbox()]

    print(toolboxes)

    print("Preprocessing...")
    # Collect all positions of separate classifications

    if len(num_of_samples) == 2 and not dock_separate:
        boat_pos = []  # Positions of boat classification
        else_pos = []  # Positions of everything else

        for region in training_regions:
            for x in range(region[0], region[2] + 1):
                for y in range(region[1], region[3] + 1):
                    if sum(class_boat_image.getpixel((x, y))) > 0:
                        boat_pos.append((x, y))
                    else:
                        else_pos.append((x, y))
    else:
        boat_pos = []
        dock_pos = []
        else_pos = []

        for region in training_regions:
            for x in range(region[0], region[2] + 1):
                for y in range(region[1], region[3] + 1):
                    if sum(class_boat_image.getpixel((x, y))) > 0:
                        boat_pos.append((x, y))
                    elif sum(class_dock_image.getpixel((x, y))) > 0:
                        dock_pos.append((x, y))
                    else:
                        else_pos.append((x, y))

    print("Collected classification and non-classification positions for sampling.")

    # Collect all of the filter images
    max_filters = []
    min_filters = []
    mean_filters = []

    for i in range(1, 12, 2):
        max_filters.append(image.filter(ImageFilter.MaxFilter(i)))
        min_filters.append(image.filter(ImageFilter.MinFilter(i)))
        mean_filters.append(image.filter(ImageFilter.BoxBlur(i)))

    edge = image.filter(ImageFilter.EDGE_ENHANCE)
    edge_plus = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    emboss = image.filter(ImageFilter.EMBOSS)

    print("Created all filters for images.")
    print("Image preprocessing finished.")

    # Register filter functions with toolbox
    for toolbox in toolboxes:
        toolbox.register("meanFilter", meanFilter, meanFilters=mean_filters)
        toolbox.register("maxFilter", maxFilter, maxFilters=max_filters)
        toolbox.register("minFilter", minFilter, minFilters=min_filters)
        toolbox.register("edgeFilter", edgeFilter, edge=edge)
        toolbox.register("edgeFilterPlus", edgeFilterPlus, edgePlus=edge_plus)
        toolbox.register("embossFilter", embossFilter, emboss=emboss)
        toolbox.register("edgesFilter", edgesFilter, edges=edges_image)

    # Create the primitive set
    pset = createPrimitiveSet(toolboxes[0], grayscale)

    # Register necessary toolbox elements for GP
    for i in range(len(toolboxes)):
        toolboxes[i].register("expr", genHalfandHalfErrorFix, pset=pset, min_=min_init_size, max_=max_init_size)
        toolboxes[i].register("individual", tools.initIterate, creator.Individual, toolboxes[i].expr)
        toolboxes[i].register("population", tools.initRepeat, list, toolboxes[i].individual)
        toolboxes[i].register("compile", gp.compile, pset=pset)

        # Setting up the evaluation function is trickier
        if (dock_separate and i == 0) or not dock_separate:  # First toolbox evaluates for boats
            if len(num_of_samples) == 3:
                toolboxes[i].register("evaluate", evalClass, toolbox=toolboxes[i], positive_pos=boat_pos,
                                      negative_pos=else_pos, negative_pos_2=dock_pos, num_of_samples=num_of_samples)
            else:
                toolboxes[i].register("evaluate", evalClass, toolbox=toolboxes[i], positive_pos=boat_pos,
                                      negative_pos=else_pos + dock_pos, num_of_samples=num_of_samples)
        elif dock_separate and i == 1:  # Second toolbox evaluates for dock
            if len(num_of_samples) == 3:
                toolboxes[i].register("evaluate", evalClass, toolbox=toolboxes[i], positive_pos=dock_pos,
                                      negative_pos=else_pos, negative_pos_2=boat_pos, num_of_samples=num_of_samples)
            else:
                toolboxes[i].register("evaluate", evalClass, toolbox=toolboxes[i], positive_pos=dock_pos,
                                      negative_pos=else_pos + boat_pos, num_of_samples=num_of_samples)
        else:
            raise Exception("This should not occur.")

        if (dock_separate and i == 0) or not dock_separate:
            toolboxes[i].register("evaluateTesting", evalTesting, toolbox=toolboxes[i], image=class_boat_image,
                                  im_range=testing_region)
        elif dock_separate and i == 1:
            toolboxes[i].register("evaluateTesting", evalTesting, toolbox=toolboxes[i], image=class_dock_image,
                                  im_range=testing_region)
        else:
            raise Exception("This should not occur.")

        toolboxes[i].register("mate", crossover_method)
        toolboxes[i].register("expr_mut", gp.genFull, min_=min_mut_size, max_=max_mut_size)
        toolboxes[i].register("mutate", mutUniformFix, expr=toolboxes[i].expr_mut, pset=pset)

        toolboxes[i].decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        toolboxes[i].decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        toolboxes[i].register("select", selTournamentElitism, numOfElites=num_of_elites, tournsize=tourn_size)

    return toolboxes


def run(params):
    print("Starting up GP run with these params: " + str(params))
    # Open the standard image
    im: Image.Image = Image.open(params['standardImageFilePath']).convert(mode="RGB")

    if params['grayscale']:
        im = im.convert(mode="L")  # Standard image in grayscale

    # Open the classification images
    classBoatIm = Image.open(params['classBoatImageFilePath']).convert(mode="RGB")
    classDockIm = Image.open(params['classDockImageFilePath']).convert(mode="RGB")

    # Open the edges image
    edgesIm = Image.open(params['edgeImageFilePath']).convert(mode="RGB")

    if params['grayscale']:
        edgesIm = edgesIm.convert(mode="L")

    print("Opened standard, classification, and edge images.")

    # Initializing things for GP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = createToolbox(params['grayscale'], params['minInitSize'], params['maxInitSize'], im, classBoatIm,
                            classDockIm, edgesIm, gp.cxOnePoint, params['minMutSize'], params['maxMutSize'],
                            params['numOfElites'], params['tournSize'], params['trainingRegions'],
                            params['testingRegion'], params['numOfSamples'], params['doDockSeparate'])

    # Keep track of best solutions and logs of all GP executions
    boat_hofs = []
    # hofs: list of (genetic program, fitness, predicted, actual, performance image)
    boat_logs = []

    if params['doDockSeparate']:
        dock_hofs = []
        dock_logs = []

    print("Performing GP for boat classification.")
    # Complete a set number of GP runs
    for i in tqdm(range(params['numOfRuns'])):
        pop = toolbox[0].population(n=params['popSize'])
        hof = tools.HallOfFame(1)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", numpy.mean)
        mstats.register("std", numpy.std)
        mstats.register("min", numpy.min)
        mstats.register("max", numpy.max)

        _, log = eaSimple(pop, toolbox[0], params['crossoverP'], params['mutationP'],
                                      params['numOfGenerations'], stats=mstats,
                                      halloffame=hof, verbose=False)

        eval = toolbox[0].evaluateTesting(hof[0])
        boat_hofs.append((hof[0], eval[0], eval[1], eval[2], eval[3]))
        boat_logs.append(log)

        print("Run " + str(i) + " complete.")

    if params['doDockSeparate']:
        print("Performing GP for dock classification.")
        for i in tqdm(range(params['numOfRuns'])):
            pop = toolbox[1].population(n=params['popSize'])
            hof = tools.HallOfFame(1)

            stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
            stats_size = tools.Statistics(len)
            mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
            mstats.register("avg", numpy.mean)
            mstats.register("std", numpy.std)
            mstats.register("min", numpy.min)
            mstats.register("max", numpy.max)

            _, log = eaSimple(pop, toolbox[1], params['crossoverP'], params['mutationP'],
                                          params['numOfGenerations'], stats=mstats,
                                          halloffame=hof, verbose=False)

            eval = toolbox[1].evaluateTesting(hof[0])
            dock_hofs.append((hof[0], eval[0], eval[1], eval[2], eval[3]))
            dock_logs.append(log)

            print("Run " + str(i) + " complete.")

    # Save all logs in pickled form
    dir_name = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    os.mkdir("./" + dir_name)

    print("Now collecting data to save in /" + dir_name)
    best_boat_hof = save_information(boat_hofs, boat_logs, dir_name, "-boat")

    if params['doDockSeparate']:
        best_dock_hof = save_information(dock_hofs, dock_logs, dir_name, "-dock")
        individuals = [best_boat_hof[0], best_dock_hof[0]]
        result = evalTestingTwoClass(individuals, toolbox, classBoatIm, params['testingRegion'])
        result[3].save(fp="./" + dir_name + '/performance-shared.png')
        result[4].save(fp="./" + dir_name + '/conflict-shared.png')
        save_confusion_matrix(result[1], result[2], dir_name, '-shared')
        with open('./' + dir_name + '/info-shared.txt', 'w') as f:
            f.write('Number of shared hits: ' + str(result[0]))


if __name__ == "__main__":
    # When executing this as main, it runs and reads from parameter file.
    params = {
        "popSize": 750,
        "crossoverP": 0.9,
        "mutationP": 0.1,
        "numOfGenerations": 100,
        "numOfRuns": 15,
        "numOfElites": 1,
        "minInitSize": 2,
        "maxInitSize": 4,
        "minMutSize": 1,
        "maxMutSize": 2,
        "tournSize": 3,
        "standardImageFilePath": "images/standard.png",
        "classBoatImageFilePath": "images/classification-boats.png",
        "classDockImageFilePath": "images/classification-dock.png",
        "edgeImageFilePath": "images/edges.png",
        "grayscale": True,
        "trainingRegions": [[28, 55, 276, 233], [36, 242, 291, 430]],  # [x0, y0, x1, y1]
        "testingRegion": [996, 210, 1211, 442],  # [x0, y0, x1, y1]
        "numOfSamples": [100, 100],  # [boat samples, everything else, dock samples (optional)]
        "doDockSeparate": False,
    }

    """
        If there are multiple training regions, then numOfGenerations will be split across the number of
        training regions. Ex. If numOfGenerations = 100 and there are 2 training regions, then 50 generations
        are spent on first, and 50 on second (kind of like transfer learning).
        
        If doDockSeparate is true, then only two values need to be in numOfSamples. It works as follows:
        -> classifications of the dock are 
    """

    try:
        f = open('params.json', 'r')
        params = json.load(f)
        f.close()
    except FileNotFoundError:
        f = open('params.json', 'w')
        json.dump(params, f)
        f.close()

    run(params)
