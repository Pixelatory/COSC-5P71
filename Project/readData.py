import pickle

import numpy as np
from matplotlib import pyplot as plt

# First set of experiments
from scipy.stats import shapiro, ttest_ind

with open('2022-04-13 11-07-39/logs-boat.pkl', 'rb') as f:
    set_1_boat = pickle.load(f)
    set_1_boat_avg = np.average([k.chapters['fitness'].select('avg') for k in set_1_boat], axis=0).tolist()
    set_1_boat_max = np.average([k.chapters['fitness'].select('max') for k in set_1_boat], axis=0).tolist()

with open('2022-04-13 06-42-27/logs-boat.pkl', 'rb') as f:
    set_2_boat = pickle.load(f)
    set_2_boat_avg = np.average([k.chapters['fitness'].select('avg') for k in set_2_boat], axis=0).tolist()
    set_2_boat_max = np.max([k.chapters['fitness'].select('max') for k in set_2_boat], axis=0).tolist()

with open('2022-04-13 11-07-39/hofs-boat.pkl', 'rb') as f:
    set_1_testing = pickle.load(f)
    set_1_testing = [k[1] for k in set_1_testing]

with open('2022-04-13 06-42-27/hofs-boat.pkl', 'rb') as f:
    set_2_testing = pickle.load(f)
    set_2_testing = [k[1] for k in set_2_testing]

# Do statistical testing

print(shapiro(set_1_testing))  # Normal
print(shapiro(set_2_testing))  # Normal

print(ttest_ind(set_2_testing, set_1_testing))  # set 2 > set 1 statistically

print(sum(set_1_testing) / len(set_1_testing), sum(set_2_testing) / len(set_2_testing))
print(np.std(set_1_testing), np.std(set_2_testing))
print(set_1_testing)

plt.close('all')
plt.plot(set_1_boat[0].select("gen"), set_1_boat_avg, 'o-', label='Set 1')
plt.plot(set_2_boat[0].select("gen"), set_2_boat_avg, 'o-', label='Set 2')
plt.plot(set_1_boat[0].select("gen"), set_1_boat_max, 'o-', label='Set 1 Best')
plt.plot(set_2_boat[0].select("gen"), set_2_boat_max, 'o-', label='Set 2 Best')
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.show()

# Second set of experiments
with open('2022-04-12 23-01-27/logs-boat.pkl', 'rb') as f:
    set_3_boat = pickle.load(f)
    set_3_boat_avg = np.average([k.chapters['fitness'].select('avg') for k in set_3_boat], axis=0).tolist()

with open('2022-04-12 23-01-27/logs-dock.pkl', 'rb') as f:
    set_3_dock = pickle.load(f)
    set_3_dock_avg = np.average([k.chapters['fitness'].select('avg') for k in set_3_dock], axis=0).tolist()

with open('2022-04-12 23-42-08/logs-boat.pkl', 'rb') as f:
    set_4_boat = pickle.load(f)
    set_4_boat_avg = np.average([k.chapters['fitness'].select('avg') for k in set_4_boat], axis=0).tolist()

with open('2022-04-12 23-42-08/logs-dock.pkl', 'rb') as f:
    set_4_dock = pickle.load(f)
    set_4_dock_avg = np.average([k.chapters['fitness'].select('avg') for k in set_4_dock], axis=0).tolist()

plt.close('all')
plt.plot(set_3_boat[0].select("gen"), set_3_boat_avg, 'o-', label='Boat Set 3')
plt.plot(set_3_boat[0].select("gen"), set_3_dock_avg, 'o-', label='Dock Set 3')
plt.plot(set_4_boat[0].select("gen"), set_4_boat_avg, 'o-', label='Boat Set 4')
plt.plot(set_4_boat[0].select("gen"), set_4_dock_avg, 'o-', label='Dock Set 4')
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.show()
