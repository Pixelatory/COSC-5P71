import pickle
import matplotlib.pyplot as plt
import numpy as np


with open('logs-1.pkl', 'rb') as f:
    test1 = pickle.load(f)

with open('logs-1.pkl', 'rb') as f:
    test3 = pickle.load(f)

with open('logs-2.pkl', 'rb') as f:
    test2 = pickle.load(f)

with open('logs-3.pkl', 'rb') as f:
    test4 = pickle.load(f)

test1Avg = np.array([k.chapters['fitness'].select('avg') for k in test1])
test1Avg = np.average(test1Avg, axis=0).tolist()

test2Avg = np.array([k.chapters['fitness'].select('avg') for k in test2])
test2Avg = np.average(test2Avg, axis=0).tolist()

test3Avg = np.array([k.chapters['fitness'].select('avg') for k in test3])
test3Avg = np.average(test3Avg, axis=0).tolist()

test4Avg = np.array([k.chapters['fitness'].select('avg') for k in test4])
test4Avg = np.average(test4Avg, axis=0).tolist()

print(test2Avg[0], test2Avg[40])


test1Min = np.array([k.chapters['fitness'].select('min') for k in test1])
test1Min = np.average(test1Min, axis=0).tolist()

test2Min = np.array([k.chapters['fitness'].select('min') for k in test2])
test2Min = np.average(test2Min, axis=0).tolist()

test3Min = np.array([k.chapters['fitness'].select('min') for k in test3])
test3Min = np.average(test3Min, axis=0).tolist()

test4Min = np.array([k.chapters['fitness'].select('min') for k in test4])
test4Min = np.average(test4Min, axis=0).tolist()

plt.close('all')
plt.plot(test1[0].select("gen"), test1Avg, 'o-', label='Set 1 Average')
plt.plot(test1[0].select("gen"), test2Avg, 'o-', label='Set 2 Average')
plt.plot(test1[0].select("gen"), test1Min, 'o-', label='Set 1 Best')
plt.plot(test1[0].select("gen"), test2Min, 'o-', label='Set 2 Best')
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.ylim([0,20])
plt.legend()
plt.show()

plt.plot(test1[0].select("gen"), test3Avg, 'o-', label='Set 3 Average')
plt.plot(test1[0].select("gen"), test4Avg, 'o-', label='Set 4 Average')
plt.plot(test1[0].select("gen"), test3Min, 'o-', label='Set 3 Best')
plt.plot(test1[0].select("gen"), test4Min, 'o-', label='Set 4 Best')
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.ylim([0,20])
plt.legend()
plt.show()

