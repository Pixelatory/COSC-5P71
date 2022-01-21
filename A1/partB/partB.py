import json
import math
import operator
import random

from deap import gp, creator, base, tools

from A1.partA.partA import protectedDiv, selTournamentElitism
from readData import read


# Define new functions
def evalClass(individual, classifications, arguments, toolbox):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    hits = 0
    for i in range(len(classifications)):
        val = func(*arguments[i])
        if val >= 0 and classifications[i] == 'M':
            hits += 1
        elif val < 0 and classifications[i] == 'B':
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


def splitArguments(data):
    """
    Splits data by argument and answer.

    :param data: Data to be split.
    :return: Tuple(List of answers, List of arguments).
    """
    answer = []
    arguments = []
    for x in data:
        answer.append(x[0])
        arguments.append(x[1:])
    return answer, arguments


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

    pset = gp.PrimitiveSetTyped("MAIN", numOfArgs, float)
    pset.addPrimitive(operator.add, [float, float], float, "add")
    pset.addPrimitive(operator.sub, [float, float], float, 'sub')
    pset.addPrimitive(operator.mul, [float, float], float, 'mul')
    pset.addPrimitive(operator.pow, [float, float], float, 'pow')
    pset.addPrimitive(protectedDiv, [float, float], float, 'div')
    pset.addPrimitive(operator.neg, [float], float, 'neg')
    pset.addPrimitive(math.cos, [float], float, 'cos')
    pset.addPrimitive(math.sin, [float], float, 'sin')
    pset.addPrimitive(max, [float], float, 'max')
    pset.addPrimitive(min, [float], float, 'min')
    pset.addPrimitive(if_then_else, [bool, float, float], float, 'if')
    pset.addPrimitive(operator.eq, [float, float], bool, 'eq')
    pset.addPrimitive(operator.gt, [float, float], bool, 'gt')
    pset.addPrimitive(operator.ge, [float, float], bool, 'gte')
    pset.addPrimitive(operator.le, [float, float], bool, 'lte')
    pset.addPrimitive(operator.lt, [float, float], bool, 'lt')
    pset.addPrimitive(operator.and_, [bool, bool], bool, 'and')
    pset.addPrimitive(operator.or_, [bool, bool], bool, 'or')
    pset.addPrimitive(operator.not_, [bool], bool, 'not')
    pset.addEphemeralConstant("rand101", lambda: float(random.uniform(-5, 5)), float)

    return pset


def createToolbox(pset, min_init_size, max_init_size, arguments, classifications, crossoverMethod, min_mut_size,
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
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=min_init_size, max_=max_init_size)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("evaluate", evalClass, toolbox=toolbox, arguments=arguments, classifications=classifications)
    toolbox.register("mate", crossoverMethod)
    toolbox.register("expr_mut", gp.genFull, min_=min_mut_size, max_=max_mut_size)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.register("select", selTournamentElitism, numOfElites=numOfElites, tournsize=tournSize)


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

    trainAnswers, trainArguments = splitArguments(trainingSet)

    pset = createPrimitiveSet(len(data[0]) - 1)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = createToolbox(pset, params['minInitSize'], params['maxInitSize'], trainArguments, trainAnswers,
                            gp.cxOnePoint, params['minMutSize'], params['maxMutSize'], params['numOfElites'],
                            params['tournSize'])
