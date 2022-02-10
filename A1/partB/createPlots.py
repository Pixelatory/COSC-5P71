import pickle

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import shapiro, kruskal, ttest_ind, mannwhitneyu
from statsmodels.graphics.gofplots import qqplot

with open('outputs/all.pkl', 'rb') as f:
    all = pickle.load(f)
    allAvg = np.array([k.chapters['fitness'].select('avg') for k in all])
    allAvg = np.average(allAvg, axis=0).tolist()
    allMax = np.array([k.chapters['fitness'].select('max') for k in all])
    allMax = np.average(allMax, axis=0).tolist()
    allAvgSize = np.array([k.chapters['size'].select('avg') for k in all])
    allAvgSize = np.average(allAvgSize, axis=0).tolist()

with open('outputs/largest.pkl', 'rb') as f:
    largest = pickle.load(f)
    largestAvg = np.array([k.chapters['fitness'].select('avg') for k in largest])
    largestAvg = np.average(largestAvg, axis=0).tolist()
    largestMax = np.array([k.chapters['fitness'].select('max') for k in largest])
    largestMax = np.average(largestMax, axis=0).tolist()
    largestAvgSize = np.array([k.chapters['size'].select('avg') for k in largest])
    largestAvgSize = np.average(largestAvgSize, axis=0).tolist()

with open('outputs/largest-stdev.pkl', 'rb') as f:
    largestStdev = pickle.load(f)
    largestStdevAvg = np.array([k.chapters['fitness'].select('avg') for k in largestStdev])
    largestStdevAvg = np.average(largestStdevAvg, axis=0).tolist()
    largestStdevMax = np.array([k.chapters['fitness'].select('max') for k in largestStdev])
    largestStdevMax = np.average(largestStdevMax, axis=0).tolist()
    largestStdevAvgSize = np.array([k.chapters['size'].select('avg') for k in largestStdev])
    largestStdevAvgSize = np.average(largestStdevAvgSize, axis=0).tolist()

with open('outputs/mean.pkl', 'rb') as f:
    mean = pickle.load(f)
    meanAvg = np.array([k.chapters['fitness'].select('avg') for k in mean])
    meanAvg = np.average(meanAvg, axis=0).tolist()
    meanMax = np.array([k.chapters['fitness'].select('max') for k in mean])
    meanMax = np.average(meanMax, axis=0).tolist()
    meanAvgSize = np.array([k.chapters['size'].select('avg') for k in mean])
    meanAvgSize = np.average(meanAvgSize, axis=0).tolist()

with open('outputs/mean-largest.pkl', 'rb') as f:
    meanLargest = pickle.load(f)
    meanLargestAvg = np.array([k.chapters['fitness'].select('avg') for k in meanLargest])
    meanLargestAvg = np.average(meanLargestAvg, axis=0).tolist()
    meanLargestMax = np.array([k.chapters['fitness'].select('max') for k in meanLargest])
    meanLargestMax = np.average(meanLargestMax, axis=0).tolist()
    meanLargestAvgSize = np.array([k.chapters['size'].select('avg') for k in meanLargest])
    meanLargestAvgSize = np.average(meanLargestAvgSize, axis=0).tolist()

with open('outputs/mean-stdev.pkl', 'rb') as f:
    meanStdev = pickle.load(f)
    meanStdevAvg = np.array([k.chapters['fitness'].select('avg') for k in meanStdev])
    meanStdevAvg = np.average(meanStdevAvg, axis=0).tolist()
    meanStdevMax = np.array([k.chapters['fitness'].select('max') for k in meanStdev])
    meanStdevMax = np.average(meanStdevMax, axis=0).tolist()
    meanStdevAvgSize = np.array([k.chapters['size'].select('avg') for k in meanStdev])
    meanStdevAvgSize = np.average(meanStdevAvgSize, axis=0).tolist()

with open('outputs/stdev.pkl', 'rb') as f:
    stdev = pickle.load(f)
    stdevAvg = np.array([k.chapters['fitness'].select('avg') for k in stdev])
    stdevAvg = np.average(stdevAvg, axis=0).tolist()
    stdevMax = np.array([k.chapters['fitness'].select('max') for k in stdev])
    stdevMax = np.average(stdevMax, axis=0).tolist()
    stdevAvgSize = np.array([k.chapters['size'].select('avg') for k in stdev])
    stdevAvgSize = np.average(stdevAvgSize, axis=0).tolist()

# This is where margins were checked
#print(abs(np.average(meanAvg) - np.average(stdevAvg)))
#print(abs(np.average(largestAvg) - np.average(stdevAvg)))

# This is where qq plots are made for averages, and shapiro-wilk test used
arrs = [(all, 'all'), (largest, 'worst'), (largestStdev, 'worst-stdev'), (mean, 'mean'), (meanLargest, 'mean-worst'),
        (meanStdev, 'mean-stdev'), (stdev, 'stdev')]

def calculateNormals(arrs, selectionStr, showPlot):
    normals = []
    for arr in arrs:
        t = np.array([k.chapters['fitness'].select(selectionStr) for k in arr[0]])
        tt = []
        for x in t:
            tt.append(x[len(x) - 1])
        if showPlot:
            qqplot(np.array(tt), line='s')
            plt.title = arr[1]
            plt.show()

        isNormal = shapiro(tt)[1] > 0.05
        normals.append(isNormal)
    return normals

normalsAvg = calculateNormals(arrs, 'avg', False)
# all not normal
# largest not normal
# largestStdev not normal
# mean is normal
# meanlargest is normal
# meanStdev not normal
# stdev is normal

# Calculating Kruskal-Wallis H-test for averages
def calculateKruskal(arrs, selectionStr):
    tt = []
    for arr in arrs:
        t = np.array([k.chapters['fitness'].select(selectionStr) for k in arr[0]])
        tt.append([])
        for x in t:
            tt[len(tt)-1].append(x[len(x) - 1])
    return kruskal(*tt)[1], tt

def createStatMatrix(arr, normals):
    # arr == tt
    hyp = np.zeros((len(arr), len(arr)), dtype=bool)
    for i in range(len(arr)):
        for j in range(len(arr)):
            if i != j:
                # If either aren't normal
                if not normals[i] or not normals[j]:
                    hyp[i][j] = mannwhitneyu(arr[i], arr[j], alternative='greater')[1] <= 0.05
                else:
                    hyp[i][j] = ttest_ind(arr[i], arr[j], alternative='greater')[1] <= 0.05
    return hyp

kr, tt = calculateKruskal(arrs, 'avg')
print(createStatMatrix(tt, normalsAvg))

normalsElite = calculateNormals(arrs, 'max', False)
kr, tt = calculateKruskal(arrs, 'max')
print(kr)
print(createStatMatrix(tt, normalsElite))

# Plots for average generational fitness, and elite
plt.close('all')
plt.plot(all[0].select("gen"), allAvg, 'o-', label='All')
plt.plot(all[0].select("gen"), largestAvg, 'o-', label='Worst')
plt.plot(all[0].select("gen"), largestStdevAvg, 'o-', label='Worst-St. Dev.')
plt.plot(all[0].select("gen"), meanAvg, 'o-', label='Mean')
plt.plot(all[0].select("gen"), meanLargestAvg, 'o-', label='Mean-Worst')
plt.plot(all[0].select("gen"), meanStdevAvg, 'o-', label='Mean-St. Dev.')
plt.plot(all[0].select("gen"), stdevAvg, 'o-', label='St. Dev.')
plt.xlabel("Generation")
plt.ylabel("Fitness")
#plt.yscale('log')
plt.legend()
plt.savefig('averages.png')
#plt.show()

#plt.close('all')
plt.plot(all[0].select("gen"), allMax, 'o-', label='All')
plt.plot(all[0].select("gen"), largestMax, 'o-', label='Worst')
plt.plot(all[0].select("gen"), largestStdevMax, 'o-', label='Worst-St. Dev.')
plt.plot(all[0].select("gen"), meanMax, 'o-', label='Mean')
plt.plot(all[0].select("gen"), meanLargestMax, 'o-', label='Mean-Worst')
plt.plot(all[0].select("gen"), meanStdevMax, 'o-', label='Mean-St. Dev.')
plt.plot(all[0].select("gen"), stdevMax, 'o-', label='St. Dev.')
plt.xlabel("Generation")
plt.ylabel("Fitness")
#plt.yscale('log')
plt.legend()
plt.savefig('bests.png')
#plt.show()

# Plot for average size of individuals per generation
#plt.close('all')
plt.plot(all[0].select("gen"), allAvgSize, 'o-', label='All')
plt.plot(all[0].select("gen"), largestAvgSize, 'o-', label='Worst')
plt.plot(all[0].select("gen"), largestStdevAvgSize, 'o-', label='Worst-St. Dev.')
plt.plot(all[0].select("gen"), meanAvgSize, 'o-', label='Mean')
plt.plot(all[0].select("gen"), meanLargestAvgSize, 'o-', label='Mean-Worst')
plt.plot(all[0].select("gen"), meanStdevAvgSize, 'o-', label='Mean-St. Dev.')
plt.plot(all[0].select("gen"), stdevAvgSize, 'o-', label='St. Dev.')
plt.xlabel("Generation")
plt.ylabel("Size")
#plt.yscale('log')
plt.legend()
plt.savefig('size_averages.png')
#plt.show()

# Plot showing the average fitness plotted with average size
plt.close()
plt.plot(allAvg, allAvgSize, 'o-', label='All')
plt.plot(largestAvg, largestAvgSize, 'o-', label='Worst')
plt.plot(largestStdevAvg, largestStdevAvgSize, 'o-', label='Worst-St. Dev.')
plt.plot(meanAvg, meanAvgSize, 'o-', label='Mean')
plt.plot(meanLargestAvg, meanLargestAvgSize, 'o-', label='Mean-Worst')
plt.plot(meanStdevAvg, meanStdevAvgSize, 'o-', label='Mean-St. Dev.')
plt.plot(stdevAvg, stdevAvgSize, 'o-', label='St. Dev.')
plt.xlabel('Fitness')
plt.ylabel('Size')
plt.legend()
plt.savefig('avg_size_and_fitness.png')
#plt.show()