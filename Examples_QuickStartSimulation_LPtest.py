###############################################################################

# Examples_QuickStartSimulation.py
# This document contains examples of how to use the simulation code.

# To launch the examples, either
# (1) Run the following command in the OS command line or Terminal:
#   python3 Examples_QuickStartSimulation.py
# (2) Copy and paste the code of this document into an interactive
#   Python console.

# Note that if modules cannot be found, this is a path problem.
# In both fixes below, replace <NAME_OF_YOUR_DIRECTORY> with a string
#   containing the directory in which the modules reside.
# (1) The following path can be updated via the OS command line or
#   Terminal (e.g. on a cluster):
#   export PYTHONPATH=<NAME_OF_YOUR_DIRECTORY>:$PYTHONPATH
# (2) The following code can be added to the main .py script file
#   (e.g. can be added to the top of this document):
#   import sys
#   sys.path.append(<NAME_OF_YOUR_DIRECTORY>)

###############################################################################

# Import other code modules
# SimObjects contains classes of objects that change within a simulation
#   replication.
# DataObjects (generally) contains classes of objects that do not change
#   within simulation replications and also do not change across
#   simulation replications -- they contain data that are "fixed"
#   for an overall problem. (Note: EpiParams is an exception -- it is
#   kind of a hybrid.)
# SimModel contains the class SimModel, which runs
#   a simulation replication and stores the simulation data
#   in its object attributes..

from Engine_SimObjects import MultiTierPolicy
from Engine_DataObjects import DataPrepConfig, TierInfo
from Engine_SimModel import SimModelConstructor

import numpy as np
import json

###############################################################################

import time

start_time = time.time()

DataPrepConfig.base_path = DataPrepConfig.base_path / "austin_data"

austin_dict_filenames = json.load(open(DataPrepConfig.base_path / "filenames.json"))

sim_model_constructor = SimModelConstructor("austin",
                                            "02/28/20",
                                            945,
                                            austin_dict_filenames)
austin_model = sim_model_constructor.create_sim_model()

tiers = TierInfo(austin_dict_filenames["tier_info_json"])

thresholds = (-1, 100, 200, 500, 1000)
mtp = MultiTierPolicy(tiers, thresholds)

austin_model.policy = mtp

austin_model.simulate_time_period(945)

print(austin_model.compute_rsq())
print(np.sum(austin_model.ICU_history))

print(time.time() - start_time)

breakpoint()