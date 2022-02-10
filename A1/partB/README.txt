This is for part B of assignment 1.

The readData.py file is where data is read from the data file
(data files are in the ./data directory). The main function has
a parameter where I could read specific subsets of data. This
was useful because that was basically my entire experimentation.

createPlots.py is where the line plots are created from the output
of GP execution. Keep in mind this isn't where the confusion matrix
or tree of best performing program is created, that's in partB.py
(at the end of the run() function).

partB.py is where the GP system is implemented. I segregated many
functions in comparison with part A's code, just for modularity.
It also provides a framework for what I'll be using when it comes
time to A2 and the final project.

The test.py file was originally for testing features, but now I
used it for executing the GP system. So when trying to execute
my code, just run test.py. You can also change the parameters in
the dictionary there.

Python version: 3.9
DEAP version: 1.3.1
networkx version: 2.6.3
matplotlib version: 3.5.1