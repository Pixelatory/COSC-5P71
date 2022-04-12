import math

from PIL import Image
from deap import gp
from deap.tools import selTournament, selBest
import random
from overload import overload


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


def doubleTournament(individuals, k, pool_size, tourn_size, num_of_elites, fit_attr_one="size", fit_attr_two="fitness",
                     fit_attr_elite="fitness"):
    """
    A first tournament selection is completed to obtain some pool_size of individuals.
    Then, another tournament selection is done on the pool of individuals gathered
    from the first set of tournaments.

    Elitism is included but can be unused just from setting num_of_elites to 0.

    :param individuals:
    :param k:
    :param pool_size:
    :param tourn_size:
    :param num_of_elites:
    :param fit_attr_one:
    :param fit_attr_two:
    :param fit_attr_elite:
    :return:
    """
    chosen = selBest(individuals, num_of_elites, fit_attr_elite)
    for i in range(k - num_of_elites):
        tmp = []
        for j in range(pool_size):
            tmp += selTournament(individuals, 1, tourn_size, fit_attr_one)
        chosen += selTournament(tmp, 1, tourn_size, fit_attr_two)
    return chosen


def evalSymbReg(individual, points, toolbox):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    sqerrors = ((func(x) - x ** 4 - x ** 3 - x ** 2 - x) ** 2 for x in points)
    return math.fsum(sqerrors) / len(points),


def evalSymbRegSize(individual, points, toolbox):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    sqerrors = ((func(x) - x ** 4 - x ** 3 - x ** 2 - x) ** 2 for x in points)
    return (math.fsum(sqerrors) / len(points)) * len(individual),


def evalSymbRegSizeAdd(individual, points, toolbox, weight):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    sqerrors = ((func(x) - x ** 4 - x ** 3 - x ** 2 - x) ** 2 for x in points)
    return (math.fsum(sqerrors) / len(points)) + (weight * len(individual)),


# Global x and y pos variables for location on the image.

xPos = 0
yPos = 0

'''
Functions are defined here.
'''


def genHalfandHalfErrorFix(pset, min_, max_, type_=None):
    """
    A fix for DEAP generation.
    """
    while True:
        try:
            return gp.genHalfAndHalf(pset, min_, max_, type_)
        except Exception:
            pass


def mutUniformFix(individual, expr, pset):
    """
    A fix for DEAP mutation.
    """
    while True:
        try:
            return gp.mutUniform(individual, expr, pset)
        except Exception:
            pass


def meanFilter(size, xOffset, yOffset, meanFilters):
    """
    Gets pixel of box blur filtered image at global vars xPos and yPos.

    :param size: Size of mean filter to use
    :type size: FilterSizes
    :type meanFilters: list[PIL.Image.Image]
    :type xOffset: int
    :type yOffset: int
    :rtype: int
    """
    size = size.value // 2
    imSize = meanFilters[size].size

    x = xPos + xOffset
    y = yPos + yOffset

    if x < 0:
        x = 0
    elif x >= imSize[0]:
        x = imSize[0] - 1

    if y < 0:
        y = 0
    elif y >= imSize[1]:
        y = imSize[1] - 1

    result = meanFilters[size].getpixel((x, y))
    if type(result) is int:  # grayscale image
        return result
    else:  # RGB image
        return Tuple(result)


def maxFilter(size, xOffset, yOffset, maxFilters):
    """
    Gets pixel of max filtered image at global vars xPos and yPos.

    :param size: Size of max filter to use
    :type size: FilterSizes
    :type maxFilters: list[PIL.Image.Image]
    :type xOffset: int
    :type yOffset: int
    :rtype: int
    """
    size = size.value // 2
    imSize = maxFilters[size].size

    x = xPos + xOffset
    y = yPos + yOffset

    if x < 0:
        x = 0
    elif x >= imSize[0]:
        x = imSize[0] - 1

    if y < 0:
        y = 0
    elif y >= imSize[1]:
        y = imSize[1] - 1

    result = maxFilters[size].getpixel((x, y))
    if type(result) is int:  # grayscale image
        return result
    else:  # RGB image
        return Tuple(result)


def minFilter(size, xOffset, yOffset, minFilters):
    """
    Gets pixel of min filtered image at global vars xPos and yPos.

    :param size: Size of min filter to use
    :type size: FilterSizes
    :type minFilters: list[PIL.Image.Image]
    :type xOffset: int
    :type yOffset: int
    :rtype: int
    """
    size = size.value // 2
    imSize = minFilters[size].size

    x = xPos + xOffset
    y = yPos + yOffset

    if x < 0:
        x = 0
    elif x >= imSize[0]:
        x = imSize[0] - 1

    if y < 0:
        y = 0
    elif y >= imSize[1]:
        y = imSize[1] - 1

    result = minFilters[size].getpixel((x, y))
    if type(result) is int:  # grayscale image
        return result
    else:  # RGB image
        return Tuple(result)


def edgeFilter(xOffset, yOffset, edge):
    """
    Gets pixel of edge filtered image at global vars xPos and yPos.

    :type edge: PIL.Image.Image
    :type xOffset: int
    :type yOffset: int
    :rtype: int
    """
    imSize = edge.size

    x = xPos + xOffset
    y = yPos + yOffset

    if x < 0:
        x = 0
    elif x >= imSize[0]:
        x = imSize[0] - 1

    if y < 0:
        y = 0
    elif y >= imSize[1]:
        y = imSize[1] - 1

    result = edge.getpixel((x, y))
    if type(result) is int:  # grayscale image
        return result
    else:  # RGB image
        return Tuple(result)


def edgeFilterPlus(xOffset, yOffset, edgePlus):
    """
    Gets pixel of edge-plus filtered image at global vars xPos and yPos,
    with offsets for x and y applied.

    :type edgePlus: PIL.Image.Image
    :type xOffset: int
    :type yOffset: int
    :rtype: int
    """
    imSize = edgePlus.size

    x = xPos + xOffset
    y = yPos + yOffset

    if x < 0:
        x = 0
    elif x >= imSize[0]:
        x = imSize[0] - 1

    if y < 0:
        y = 0
    elif y >= imSize[1]:
        y = imSize[1] - 1

    result = edgePlus.getpixel((x, y))
    if type(result) is int:  # grayscale image
        return result
    else:  # RGB image
        return Tuple(result)


def edgesFilter(xOffset, yOffset, edges):
    """
    Gets pixel of edge filtered image at global vars xPos and yPos.

    :type edges: PIL.Image.Image
    :type xOffset: int
    :type yOffset: int
    :rtype: int
    """
    imSize = edges.size

    x = xPos + xOffset
    y = yPos + yOffset

    if x < 0:
        x = 0
    elif x >= imSize[0]:
        x = imSize[0] - 1

    if y < 0:
        y = 0
    elif y >= imSize[1]:
        y = imSize[1] - 1

    result = edges.getpixel((x, y))
    if type(result) is int:  # grayscale image
        return result
    else:  # RGB image
        return Tuple(result)


def embossFilter(xOffset, yOffset, emboss):
    """
    Gets pixel of embossed image at global vars xPos and yPos.

    :type emboss: PIL.Image.Image
    :type xOffset: int
    :type yOffset: int
    :rtype: int
    """
    imSize = emboss.size

    x = xPos + xOffset
    y = yPos + yOffset

    if x < 0:
        x = 0
    elif x >= imSize[0]:
        x = imSize[0] - 1

    if y < 0:
        y = 0
    elif y >= imSize[1]:
        y = imSize[1] - 1

    result = emboss.getpixel((x, y))
    if type(result) is int:  # grayscale image
        return result
    else:  # RGB image
        return Tuple(result)


def protectedDiv(left, right):
    """
    Don't allow for division by 0 (just return 1 in that case).

    :param left: numerator
    :param right: denominator
    :return: left / right
    """
    try:
        return left / right
    except ZeroDivisionError:
        return 1.0


def intensity(val):
    """
    Gets the average of red, green, and blue channels.

    :type val: Tuple
    """
    return (val.get(0) + val.get(1) + val.get(2)) / 3


def redChannel(val):
    """
    Gets the red channel value.

    :type val: Tuple
    """
    return val.get(0)


def blueChannel(val):
    """
    Gets the blue channel value.

    :type val: Tuple
    """
    return val.get(1)


def greenChannel(val):
    """
    Gets the green channel value.

    :type val: Tuple
    """
    return val.get(2)


def evalClass(individual, toolbox, num_of_samples, positive_pos, negative_pos, negative_pos_2=None):
    """
    Evaluate programs during training.

    :param individual: Individual to be evaluated
    :type toolbox: deap.base.Toolbox
    :param positive_pos: Positions of positive classification pixels
    :type positive_pos: list[tuple[int, int]]
    :param negative_pos: Positions of pixels which are negative
    :type negative_pos: list[tuple[int, int]]
    :param negative_pos_2: Positions of pixels which are negative, but should be separate from negative_pos
    :type negative_pos_2: list[tuple[int, int]]
    :param num_of_samples: Number of samples to take of each classPos and nonClassPos
    :type num_of_samples: tuple[int, int] or tuple[int, int, int]
    """
    if len(num_of_samples) == 3 and negative_pos_2 is None:
        raise Exception()
    global xPos, yPos
    hits = 0

    # First, sample from classifications and non-classifications
    class_pos = random.sample(positive_pos, num_of_samples[0])
    non_class_pos = random.sample(negative_pos, num_of_samples[1])

    if len(num_of_samples) == 3 and negative_pos_2 is not None:
        non_class_pos_2 = random.sample(negative_pos_2, num_of_samples[2])

    # Calculate hits for true classifications
    for pos in class_pos:
        xPos = pos[0]
        yPos = pos[1]
        prediction = toolbox.compile(expr=individual) >= 0

        if prediction:
            hits += 1

    # Calculate hits for false classifications
    for pos in non_class_pos:
        xPos = pos[0]
        yPos = pos[1]
        prediction = toolbox.compile(expr=individual) >= 0

        if not prediction:
            hits += 1

    if len(num_of_samples) == 3 and negative_pos_2 is not None:
        for pos in non_class_pos_2:
            xPos = pos[0]
            yPos = pos[1]
            prediction = toolbox.compile(expr=individual) >= 0

            if not prediction:
                hits += 1
    return hits,


def evalClassDivSize(individual, toolbox, classPos, nonClassPos, numOfSamples):
    """
    Evaluate the program during training, but in this case the fitness is divided by the length of individual.

    :param individual: Individual to be evaluated
    :type toolbox: deap.base.Toolbox
    :param classPos: Positions of true classified pixels
    :type classPos: list[tuple[int, int]]
    :param nonClassPos: Positions of non-classified pixels
    :type nonClassPos: list[tuple[int, int]]
    :param numOfSamples: Number of samples to take of each classPos and nonClassPos
    :type numOfSamples: tuple[int, int]
    """
    return evalClass(individual, toolbox, classPos, nonClassPos, numOfSamples)[0] / len(individual),


def evalClassSubSize(individual, toolbox, classPos, nonClassPos, numOfSamples, weight):
    """
    Evaluate the program during training, but in this case the fitness is subtracted by the size of individual times
    a provided weight value.

    :param individual: Individual to be evaluated
    :type toolbox: deap.base.Toolbox
    :param classPos: Positions of true classified pixels
    :type classPos: list[tuple[int, int]]
    :param nonClassPos: Positions of non-classified pixels
    :type nonClassPos: list[tuple[int, int]]
    :param numOfSamples: Number of samples to take of each classPos and nonClassPos
    :type numOfSamples: tuple[int, int]
    """
    return evalClass(individual, toolbox, classPos, nonClassPos, numOfSamples)[0] - (weight * len(individual)),


def evalTesting(individual, toolbox, image, im_range):
    """
    Evaluate programs during testing.

    :param individual: Individual to be evaluated
    :type toolbox: deap.base.Toolbox
    :param image: The classification image
    :type image: Image.Image
    :type im_range: list[int, int, int, int]
    :return: (hits, predicted, actual, performance image)
    :rtype: tuple[int, list, list, Image.Image]
    """
    global xPos, yPos
    # Transform the tree expression in a callable function
    x0, y0, x1, y1 = im_range[0], im_range[1], im_range[2], im_range[3]
    pImage = Image.new(mode="RGB", size=(x1 - x0 + 1, y1 - y0 + 1))  # The performance image

    predicted = []
    actual = []
    hits = 0
    for x in range(x0, x1 + 1):
        xPos = x
        for y in range(y0, y1 + 1):
            yPos = y
            classification = sum(image.getpixel((xPos, yPos))) > 0  # Classification image has some sort of colour here
            prediction = toolbox.compile(expr=individual) >= 0

            if (prediction and classification) or \
                    (not prediction and not classification):
                hits += 1

            # Draw the performance image
            if prediction and classification:  # True positive
                pImage.putpixel((x - x0, y - y0), (0, 255, 0))
            elif not prediction and not classification:  # True negative
                pImage.putpixel((x - x0, y - y0), (0, 0, 0))
            elif not prediction and classification:  # False Negative
                pImage.putpixel((x - x0, y - y0), (255, 255, 0))
            elif prediction and not classification:  # False Positive
                pImage.putpixel((x - x0, y - y0), (255, 0, 0))

            actual.append(classification)
            predicted.append(prediction)
    return hits, predicted, actual, pImage


def evalTestingTwoClass(individuals, toolboxes, image, im_range):
    """
    Evaluate programs during testing with two different types of classifications on the same image.

    The first entry in individuals should be the true positive.

    :param individuals: Individuals to be evaluated
    :type toolboxes: list[deap.base.Toolbox]
    :param image: The classification image.
    :type image: Image.Image
    :type im_range: list[int, int, int, int]
    :return: (hits, predicted, actual, performance image, confliction image)
    :rtype: tuple[int, list, list, Image.Image, Image.Image]
    """
    global xPos, yPos
    # Transform the tree expression in a callable function
    x0, y0, x1, y1 = im_range[0], im_range[1], im_range[2], im_range[3]
    pImage = Image.new(mode="RGB", size=(x1 - x0 + 1, y1 - y0 + 1))  # Performance image
    cImage = Image.new(mode="RGB", size=(x1 - x0 + 1, y1 - y0 + 1))  # Confliction image

    predicted = []
    actual = []
    hits = 0
    for x in range(x0, x1 + 1):
        xPos = x
        for y in range(y0, y1 + 1):
            yPos = y
            # Classification images have some sort of colour here
            classification = sum(image.getpixel((xPos, yPos))) > 0
            prediction_1 = toolboxes[0].compile(expr=individuals[0]) >= 0
            prediction_2 = toolboxes[1].compile(expr=individuals[1]) < 0

            the_same = True if prediction_1 and prediction_2 else False

            prediction = the_same

            if (prediction and classification) or \
                    (not prediction and not classification):
                hits += 1

            # Draw the performance image
            if prediction and classification:  # True positive
                pImage.putpixel((x - x0, y - y0), (0, 255, 0))
            elif not prediction and not classification:  # True negative
                pImage.putpixel((x - x0, y - y0), (0, 0, 0))
            elif not prediction and classification:  # False Negative
                pImage.putpixel((x - x0, y - y0), (255, 255, 0))
            elif prediction and not classification:  # False Positive
                pImage.putpixel((x - x0, y - y0), (255, 0, 0))

            if the_same:
                cImage.putpixel((x - x0, y - y0), (0, 0, 0))
            else:
                cImage.putpixel((x - x0, y - y0), (255, 255, 255))

            actual.append(classification)
            predicted.append(prediction)
    return hits, predicted, actual, pImage, cImage


class Tuple:
    @overload
    def __init__(self, val: int):
        self.capacity = val
        self.items = []

    @__init__.add
    def __init__(self, val: list):
        self.capacity = len(val)
        self.items = val

    @__init__.add
    def __init__(self, val: tuple):
        self.capacity = len(val)
        self.items = [x for x in val]

    def append(self, item):
        if self.capacity < len(self.items):
            self.items.append(item)
            self.capacity += 1

    def get(self, idx):
        return self.items[idx]


'''
Create dummy classes so that filter sizes can only be with the specified integer values.
There might be an easier way to directly specify which integer values are allowed in
the primitive set, but this one came to me in two seconds so we'll run with it.
'''


class FilterSizes:
    value = None  # Please override me.
    # The ruler of filter size dummies.
    pass


class FilterSizeOne(FilterSizes):
    value = 1


class FilterSizeThree(FilterSizes):
    value = 3


class FilterSizeFive(FilterSizes):
    value = 5


class FilterSizeSeven(FilterSizes):
    value = 7


class FilterSizeNine(FilterSizes):
    value = 9


class FilterSizeEleven(FilterSizes):
    value = 11
