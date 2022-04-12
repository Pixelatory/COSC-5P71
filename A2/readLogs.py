import pickle
import numpy as np
from deap import gp, creator
from matplotlib import pyplot as plt
import seaborn as sn
from scipy.stats import shapiro, mannwhitneyu

'''
# Comparing sampling sizes.
logs = []
with open('./outputs/2022-02-25 22-31-48/logs.pkl', 'rb') as f:
    logs.append(pickle.load(f))
    allAvg = np.average([k.chapters['fitness'].select('avg') for k in logs[len(logs) - 1]], axis=0).tolist()

with open('./outputs/2022-02-26 11-20-08/logs.pkl', 'rb') as f:
    logs.append(pickle.load(f))
    allAvg2 = np.average([k.chapters['fitness'].select('avg') for k in logs[len(logs) - 1]], axis=0).tolist()

with open('./outputs/2022-02-26 11-36-44/logs.pkl', 'rb') as f:
    logs.append(pickle.load(f))
    allAvg3 = np.average([k.chapters['fitness'].select('avg') for k in logs[len(logs) - 1]], axis=0).tolist()

with open('./outputs/2022-02-26 12-01-49/logs.pkl', 'rb') as f:
    logs.append(pickle.load(f))
    allAvg4 = np.average([k.chapters['fitness'].select('avg') for k in logs[len(logs) - 1]], axis=0).tolist()

with open('./outputs/2022-02-26 12-29-46/logs.pkl', 'rb') as f:
    logs.append(pickle.load(f))
    allAvg5 = np.average([k.chapters['fitness'].select('avg') for k in logs[len(logs) - 1]], axis=0).tolist()

# Getting fitness for best results for all sampling sizes here.
one = [logs[0][i].chapters['fitness'].select('max')[100] for i in range(len(logs[0]))]  # normal
two = [logs[1][i].chapters['fitness'].select('max')[100] for i in range(len(logs[0]))]  # non-normal
three = [logs[2][i].chapters['fitness'].select('max')[100] for i in range(len(logs[0]))]  # non-normal
four = [logs[3][i].chapters['fitness'].select('max')[100] for i in range(len(logs[0]))]  # non-normal
five = [logs[4][i].chapters['fitness'].select('max')[100] for i in range(len(logs[0]))]  # non-normal

print(shapiro(one))
print(shapiro(two))
print(shapiro(three))
print(shapiro(four))
print(shapiro(five))

# only 1 is normal, so we have to use mann-whitney u for all anyways
print("one:")
print(mannwhitneyu(one, two, alternative='greater'))  # better
print(mannwhitneyu(one, three, alternative='greater'))  # better
print(mannwhitneyu(one, four, alternative='greater'))  # better
print(mannwhitneyu(one, five, alternative='greater'))  # better

print("two:")
print(mannwhitneyu(two, one, alternative='greater'))  # non-better
print(mannwhitneyu(two, three, alternative='greater'))  # non-better
print(mannwhitneyu(two, four, alternative='greater'))  # non-better
print(mannwhitneyu(two, five, alternative='greater'))  # non-better

print("three:")
print(mannwhitneyu(three, one, alternative='greater'))  # non-better
print(mannwhitneyu(three, two, alternative='greater'))  # better
print(mannwhitneyu(three, four, alternative='greater'))  # better
print(mannwhitneyu(three, five, alternative='greater'))  # non-better

print("four:")
print(mannwhitneyu(four, one, alternative='greater'))  # non-better
print(mannwhitneyu(four, two, alternative='greater'))  # non-better
print(mannwhitneyu(four, three, alternative='greater'))  # non-better
print(mannwhitneyu(four, five, alternative='greater'))  # non-better

print("five:")
print(mannwhitneyu(five, one, alternative='greater'))  # non-better
print(mannwhitneyu(five, two, alternative='greater'))  # better
print(mannwhitneyu(five, three, alternative='greater'))  # non-better
print(mannwhitneyu(five, four, alternative='greater'))  # better

plt.close('all')
plt.plot(logs[0][0].select("gen"), allAvg, 'o-', label='100-100')
plt.plot(logs[0][0].select("gen"), allAvg2, 'o-', label='50-10')
plt.plot(logs[0][0].select("gen"), allAvg3, 'o-', label='100-10')
plt.plot(logs[0][0].select("gen"), allAvg4, 'o-', label='10-50')
plt.plot(logs[0][0].select("gen"), allAvg5, 'o-', label='10-100')
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.show()
'''


'''
Comparing before adding emboss and new edges filter to after (unused; would've cramped up report with little findings)

logs = []
with open('2022-02-25 22-31-48/logs.pkl', 'rb') as f:
    logs.append(pickle.load(f))
    allAvg = np.average([k.chapters['fitness'].select('avg') for k in logs[len(logs) - 1]], axis=0).tolist()

with open('2022-03-01 16-10-40/logs.pkl', 'rb') as f:
    logs.append(pickle.load(f))
    allAvg2 = np.average([k.chapters['fitness'].select('avg') for k in logs[len(logs) - 1]], axis=0).tolist()

plt.close('all')
plt.plot(logs[0][0].select("gen"), allAvg, 'o-', label='Old')
plt.plot(logs[0][0].select("gen"), allAvg2, 'o-', label='New')
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.show()
'''

'''
# Compare before and after adding offsets
logs = []
with open('./outputs/2022-03-01 16-10-40/logs.pkl', 'rb') as f:  # before offsets
    logs.append(pickle.load(f))
    allAvg = np.average([k.chapters['fitness'].select('avg') for k in logs[len(logs) - 1]], axis=0).tolist()
    allMax = np.min([k.chapters['fitness'].select('max') for k in logs[len(logs) - 1]], axis=0).tolist()

with open('./outputs/2022-03-02 01-09-21/logs.pkl', 'rb') as f:  # offsets
    logs.append(pickle.load(f))
    allAvg2 = np.average([k.chapters['fitness'].select('avg') for k in logs[len(logs) - 1]], axis=0).tolist()
    allMax2 = np.min([k.chapters['fitness'].select('max') for k in logs[len(logs) - 1]], axis=0).tolist()

# Getting fitness for best results from before offsets and after
beforeOffsets = [logs[0][i].chapters['fitness'].select('max')[100] for i in range(len(logs[0]))]
afterOffsets = [logs[1][i].chapters['fitness'].select('max')[100] for i in range(len(logs[0]))]

print(shapiro(beforeOffsets))  # not normally distributed
print(shapiro(afterOffsets))  # not normally distributed

print(mannwhitneyu(beforeOffsets, afterOffsets, alternative='greater'))  # no statistical difference in median
print(mannwhitneyu(afterOffsets, beforeOffsets, alternative='greater'))  # no statistical difference in median


plt.close('all')
plt.plot(logs[0][0].select("gen"), allAvg, 'o-', label='No Offsets')
plt.plot(logs[0][0].select("gen"), allAvg2, 'o-', label='Offsets')
plt.plot(logs[0][0].select("gen"), allMax, 'o-', label='No Offsets (Best)')
plt.plot(logs[0][0].select("gen"), allMax2, 'o-', label='Offsets (Best)')
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.show()
'''

'''
# Compare RGB vs Grayscale on both images
logs = []
with open('./outputs/2022-03-10 02-29-03/logs.pkl', 'rb') as f:  # RGB Image 1
    logs.append(pickle.load(f))
    allAvg = np.average([k.chapters['fitness'].select('avg') for k in logs[len(logs) - 1]], axis=0).tolist()
    allMax = np.min([k.chapters['fitness'].select('max') for k in logs[len(logs) - 1]], axis=0).tolist()
    allSize = np.average([k.chapters['size'].select('avg') for k in logs[len(logs) - 1]], axis=0).tolist()

with open('./outputs/2022-03-09 17-53-11/logs.pkl', 'rb') as f:  # RGB Image 2
    logs.append(pickle.load(f))
    allAvg2 = np.average([k.chapters['fitness'].select('avg') for k in logs[len(logs) - 1]], axis=0).tolist()
    allMax2 = np.min([k.chapters['fitness'].select('max') for k in logs[len(logs) - 1]], axis=0).tolist()
    allSize2 = np.average([k.chapters['size'].select('avg') for k in logs[len(logs) - 1]], axis=0).tolist()

with open('./outputs/2022-03-02 01-09-21/logs.pkl', 'rb') as f:  # Grayscale Image 1
    logs.append(pickle.load(f))
    allAvg3 = np.average([k.chapters['fitness'].select('avg') for k in logs[len(logs) - 1]], axis=0).tolist()
    allMax3 = np.min([k.chapters['fitness'].select('max') for k in logs[len(logs) - 1]], axis=0).tolist()
    allSize3 = np.average([k.chapters['size'].select('avg') for k in logs[len(logs) - 1]], axis=0).tolist()

with open('./outputs/2022-03-08 18-11-14/logs.pkl', 'rb') as f:  # Grayscale Image 2
    logs.append(pickle.load(f))
    allAvg4 = np.average([k.chapters['fitness'].select('avg') for k in logs[len(logs) - 1]], axis=0).tolist()
    allMax4 = np.min([k.chapters['fitness'].select('max') for k in logs[len(logs) - 1]], axis=0).tolist()
    allSize4 = np.average([k.chapters['size'].select('avg') for k in logs[len(logs) - 1]], axis=0).tolist()


# Getting fitness for best results
rgbim1 = [logs[0][i].chapters['fitness'].select('max')[100] for i in range(len(logs[0]))]
rgbim2 = [logs[1][i].chapters['fitness'].select('max')[100] for i in range(len(logs[0]))]
grayim1 = [logs[2][i].chapters['fitness'].select('max')[100] for i in range(len(logs[0]))]
grayim2 = [logs[3][i].chapters['fitness'].select('max')[100] for i in range(len(logs[0]))]

print(shapiro(rgbim1))  # normal
print(shapiro(rgbim2))  # non-normal
print(shapiro(grayim1))  # non-normal
print(shapiro(grayim2))  # non-normal

print(mannwhitneyu(rgbim1, grayim1, alternative="greater"))  # no difference
print(mannwhitneyu(grayim1, rgbim1, alternative="greater"))  # no difference
print(mannwhitneyu(rgbim2, grayim2, alternative="greater"))  # rgb > gray on image 2
print(mannwhitneyu(grayim2, rgbim2, alternative="greater"))  # gray not > rgb on im 2



plt.close('all')
plt.plot(logs[0][0].select("gen"), allAvg, 'o-', label='RGB Image 1')
plt.plot(logs[0][0].select("gen"), allAvg2, 'o-', label='RGB Image 2')
plt.plot(logs[0][0].select("gen"), allAvg3, 'o-', label='Grayscale Image 1')
plt.plot(logs[0][0].select("gen"), allAvg4, 'o-', label='Grayscale Image 2')
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.show()


plt.close('all')
plt.plot(allAvg, allSize, 'o-', label='RGB Image 1')
plt.plot(allAvg2, allSize2, 'o-', label='RGB Image 2')
plt.plot(allAvg3, allSize3, 'o-', label='Grayscale Image 1')
plt.plot(allAvg4, allSize4, 'o-', label='Grayscale Image 2')
plt.xlabel("Fitness")
plt.ylabel("Size")
plt.legend()
plt.show()
'''

'''
# Compare population size
logs = []
with open('./outputs/2022-03-10 17-14-33/logs.pkl', 'rb') as f:  # Image 1 750 population
    logs.append(pickle.load(f))
    allAvg = np.average([k.chapters['fitness'].select('avg') for k in logs[len(logs) - 1]], axis=0).tolist()
    allMax = np.min([k.chapters['fitness'].select('max') for k in logs[len(logs) - 1]], axis=0).tolist()

with open('./outputs/2022-03-10 02-29-03/logs.pkl', 'rb') as f:  # image 1 300 pop
    logs.append(pickle.load(f))
    allAvg2 = np.average([k.chapters['fitness'].select('avg') for k in logs[len(logs) - 1]], axis=0).tolist()
    allMax2 = np.min([k.chapters['fitness'].select('max') for k in logs[len(logs) - 1]], axis=0).tolist()

with open('./outputs/2022-03-10 08-18-13/logs.pkl', 'rb') as f:  # Image 2 750 population
    logs.append(pickle.load(f))
    allAvg3 = np.average([k.chapters['fitness'].select('avg') for k in logs[len(logs) - 1]], axis=0).tolist()
    allMax3 = np.min([k.chapters['fitness'].select('max') for k in logs[len(logs) - 1]], axis=0).tolist()

with open('./outputs/2022-03-09 17-53-11/logs.pkl', 'rb') as f:  # im 2 300 pop
    logs.append(pickle.load(f))
    allAvg4 = np.average([k.chapters['fitness'].select('avg') for k in logs[len(logs) - 1]], axis=0).tolist()
    allMax4 = np.min([k.chapters['fitness'].select('max') for k in logs[len(logs) - 1]], axis=0).tolist()

# Getting fitness for best results
im1_750 = [logs[0][i].chapters['fitness'].select('max')[100] for i in range(len(logs[0]))]
im1_300 = [logs[1][i].chapters['fitness'].select('max')[100] for i in range(len(logs[0]))]
im2_750 = [logs[2][i].chapters['fitness'].select('max')[100] for i in range(len(logs[0]))]
im2_300 = [logs[3][i].chapters['fitness'].select('max')[100] for i in range(len(logs[0]))]

print(shapiro(im1_750))  # non-normal
print(shapiro(im1_300))  # non-normal
print(shapiro(im2_750))  # non-normal
print(shapiro(im2_300))  # non-normal

print(mannwhitneyu(im1_750, im1_300, alternative="greater"))
print(mannwhitneyu(im1_300, im1_750, alternative="greater"))
print(mannwhitneyu(im2_300, im2_750, alternative="greater"))
print(mannwhitneyu(im2_750, im2_750, alternative="greater"))


plt.close('all')
plt.plot(logs[0][0].select("gen"), allAvg, 'o-', label='Image 1 (750)')
plt.plot(logs[0][0].select("gen"), allAvg2, 'o-', label='Image 1 (300)')
plt.plot(logs[0][0].select("gen"), allAvg3, 'o-', label='Image 2 (750)')
plt.plot(logs[0][0].select("gen"), allAvg4, 'o-', label='Image 2 (300)')
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.show()
'''

creator.create("Individual", gp.PrimitiveTree)

with open('../Project/2022-04-09 06-45-45/hofs.pkl', 'rb') as f:
    hofs = pickle.load(f)
    print(hofs)