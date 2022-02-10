This is for part A of assignment 1.

partA.py has the DEAP system implementation, where main() has most of the code for GP.

At the end of the "if __name__ == "__main__":" condition within partA.py,
this is where I implemented the functionality for outputting a diagram of
the best individual's tree.

params.json is the parameter configuration file.

I did not include reading points from a file because there was no
need for me to, since Python is not a compiled language. Had I used
ECJ, I definitely would've included the functionality for reading
from a file.

In readData.py, this is where I read from the GP logs and created
the plots included within the report.

Python version: 3.9
DEAP version: 1.3.1
networkx version: 2.6.3
matplotlib version: 3.5.1