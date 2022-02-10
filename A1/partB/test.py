from A1.partB.partB import run

if __name__ == "__main__":
    params = {
            'popSize': 300,
            'crossoverP': 0.9,
            'mutationP': 0.1,
            'numOfGenerations': 100,
            'numOfRuns': 10,
            'numOfElites': 1,
            'trainSetPercentage': 0.8,
            'minInitSize': 2,
            'maxInitSize': 4,
            'minMutSize': 1,
            'maxMutSize': 2,
            'tournSize': 3,
            'dataReadOption': ['largest', 'stdev'],
        }

    run(params)