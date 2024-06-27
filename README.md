# SEIRModelComponents

An example Jupyter Notebook, demo_for_discussion.ipynb, has been made available as a code tutorial.

## Requirements
- Python 3 (this code was built using Python 3.4)
- numpy, pandas
- pathlib (comes automatically with Python >= 3.4)
- json (automatically included in Python)

## Overview of modules:
- SimObjects contains classes of objects that change within a simulation replication.
- DataObjects (generally) contains classes of objects that do not change
  within simulation replications and also do not change across
  simulation replications -- they contain data that are "fixed"
  for an overall problem. (Note: EpiParams is an exception -- it is
  kind of a hybrid.)
- SimModel contains the class SimModel, which runs
  a simulation replication and stores the simulation data
  in its object attributes..

## Known bugs:
- The stepsize is accidentally 9, not 10 (so that we are splitting a day into 9 steps for discretization)
- The update for S seems suspicious (potentially a typo)
- Booster doses take effect immediately, whereas the first and second vaccine dose take time to become effective
- Fixed_kappa_end_date seems to be buggy

## Things we need to do:
- Unit tests and integration tests
- Need function signatures and parameter definitions for all classes and methods
- Write least-squares (calibration), optimization, and input/output (importing/exporting intermediate simulation states) 
to be compatible with this code version
- Incorporate new automatic calibration ideas/code (from Guyi)
- Incorporate wastewater code (wastewater code updates have diverged significantly)
- Fix messy data inputs -- there are too many .csv and .json files 
and the dictionary formats are not consistent -- this leads to unreadable code in parsing/loading
the data

Note that if modules cannot be found, this is a path problem.
In both fixes below, replace <NAME_OF_YOUR_DIRECTORY> with a string
  containing the directory in which the modules reside.

(1) The following path can be updated via the OS command line or
  Terminal (e.g. on a cluster):
  export PYTHONPATH=<NAME_OF_YOUR_DIRECTORY>:$PYTHONPATH

(2) The following code can be added to the main .py script file
  (e.g. can be added to the top of this document):
  import sys
  sys.path.append(<NAME_OF_YOUR_DIRECTORY>)


