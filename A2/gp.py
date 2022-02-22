import operator
import random
from functools import partial

from PIL import Image, ImageFilter
from deap import gp, base, tools, creator

# TODO: mark boats on ground truth image.

# Define new functions
from deap.tools import selBest, selTournament

xPos = 0
yPos = 0

im = Image.open("images/standard.png")
im = im.convert(mode="L")  # Standard image in grayscale

print(im.size)
width, height = im.size
print(width, height)

maxFilters = []
minFilters = []
meanFilters = []

for i in range(1, 12, 2):
    maxFilters.append(im.filter(ImageFilter.MaxFilter(i)))
    minFilters.append(im.filter(ImageFilter.MinFilter(i)))
    meanFilters.append(im.filter(ImageFilter.BoxBlur(i)))

edge = im.filter(ImageFilter.EDGE_ENHANCE)
edgePlus = im.filter(ImageFilter.EDGE_ENHANCE_MORE)

def meanFilter(size):
    """
    Applies to mean filter on image.

    :param size: Size of mean filter to use
    :type size: int
    :rtype: int
    """
    idx = size // 2
    return meanFilters[idx].getpixel((xPos, yPos))


def maxFilter(size):
    """
    Applies to mean filter on image.

    :param size: Size of mean filter to use
    :type size: int
    :rtype: int
    """
    idx = size // 2
    return maxFilters[idx].getpixel((xPos, yPos))


def minFilter(size):
    """
    Applies to minimum filter on image.

    :param size: Size of mean filter to use
    :type size: int
    :rtype: int
    """
    idx = size // 2
    return minFilters[idx].getpixel((xPos, yPos))


def edgeFilter():
    """
    Applies the edge filter on image.
    :rtype: int
    """
    return edge.getpixel((xPos, yPos))


def edgeFilterPlus():
    """
    Applies the edgePlus filter on image.
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

def evalImage(individual, toolbox, image):
    # TODO: finish the performance image implementation
    """

    :type toolbox: deap.base.Toolbox
    :type image: PIL.Image.Image
    :return:
    :rtype: PIL.Image.Image
    """
    pImage = Image.new(mode="RGB", size=image.size)  # The performance image

def eval(individual, toolbox, image):
    # TODO: rework this to match the needs of image classification.
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    func(0, 0)

    width, height = image.size  # Tuple (width, height)

    for x in range(width):
        for y in range(height):
            pass
    '''
    hits = 0
    for x in trainingSet:
        val = func(*x[1:])
        if (val >= 0 and x[0] == 'M') \
                or (val < 0 and x[0] == 'B'):
            hits += 1
    '''
    return 0,


def createPrimitiveSet(toolbox):
    """

    :return: Primitive Set
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
    pset.addPrimitive(meanFilter, [int], float, "MEANFIL")
    pset.addPrimitive(meanFilter, [int], float, "MEANFIL")
    pset.addTerminal(1, int)
    pset.addTerminal(3, int)
    pset.addTerminal(5, int)
    pset.addTerminal(7, int)
    pset.addTerminal(9, int)
    pset.addTerminal(11, int)

    # TODO: add filters to primitive set here.
    return pset


def createToolbox(pset, min_init_size, max_init_size, image, crossoverMethod, min_mut_size,
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

    # Register image filter functions
    toolbox.register("meanFilter", meanFilter)

    # Register rest of necessary toolbox elements for GP
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=min_init_size, max_=max_init_size)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("evaluate", eval, toolbox=toolbox, image=image)
    toolbox.register("mate", crossoverMethod)
    toolbox.register("expr_mut", gp.genFull, min_=min_mut_size, max_=max_mut_size)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.register("select", selTournamentElitism, numOfElites=numOfElites, tournsize=tournSize)

    return toolbox
