import random

import torch
from PIL import Image
from deap import gp
from deap.tools import selBest, selTournament
from overload import overload

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


def evalClass(individual, toolbox, classPos, nonClassPos, numOfSamples):
    """
    Evaluate programs during training.

    :param individual: Individual to be evaluated
    :type toolbox: deap.base.Toolbox
    :param classPos: Positions of true classified pixels
    :type classPos: list[tuple[int, int]]
    :param nonClassPos: Positions of non-classified pixels
    :type nonClassPos: list[tuple[int, int]]
    :param numOfSamples: Number of samples to take of each classPos and nonClassPos
    :type numOfSamples: tuple[int, int]
    """
    global xPos, yPos
    hits = 0

    # First, sample from classifications and non-classifications
    classPos = random.sample(classPos, numOfSamples[0])
    nonClassPos = random.sample(nonClassPos, numOfSamples[1])

    # Calculate hits for true classifications
    for pos in classPos:
        xPos = pos[0]
        yPos = pos[1]
        prediction = toolbox.compile(expr=individual) >= 0

        if prediction:
            hits += 1

    # Calculate hits for false classifications
    for pos in nonClassPos:
        xPos = pos[0]
        yPos = pos[1]
        prediction = toolbox.compile(expr=individual) >= 0

        if not prediction:
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

    predicted = []
    actual = []
    hits = 0
    for x in range(xRange[0], xRange[1] + 1):
        xPos = x
        for y in range(yRange[0], yRange[1] + 1):
            yPos = y
            classification = sum(image.getpixel((xPos, yPos))) > 0  # Classification image has some sort of colour here
            prediction = toolbox.compile(expr=individual) >= 0

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


def performanceImage(pred, target):
    """
    Creates a performance image similarly to in evalTesting(). However,
    this is primarly used by SegNet.

    :param pred: array of predicted classifications
    :type pred: torch.Tensor
    :param target: array of actual classifications
    :type target: torch.Tensor
    :param coordRange: Coordinate ranges on standard classification image
    :rtype: Image.Image
    """
    # The performance image
    pImage = Image.new(mode="RGB", size=(pred.size(dim=1), pred.size(dim=0)))
    target = target * 2
    pred = torch.where(pred >= 0.5, 1.0, 0.0)
    res = (pred - target).to('cpu')
    '''
        res:
        0 = true neg
        -2 = false neg
        1 = false pos
        -1 = true pos
    '''
    for x in range(pImage.size[0]):
        for y in range(pImage.size[1]):
            if res[y][x] == 0:
                pImage.putpixel((x, y), (0, 0, 0))
            elif res[y][x] == -2:
                pImage.putpixel((x, y), (255, 255, 0))
            elif res[y][x] == 1:
                pImage.putpixel((x, y), (255, 0, 0))
            elif res[y][x] == -1:
                pImage.putpixel((x, y), (0, 255, 0))
    return pImage


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
