###############################################################################

import matplotlib.pyplot as plt

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

# Linda Pei 2023

###############################################################################

# Import other code modules
# SimObjects contains classes of objects that change within a simulation
#   replication.
# DataObjects contains classes of objects that do not change
#   within simulation replications and also do not change *across*
#   simulation replications -- they contain data that are "fixed"
#   for an overall problem.
# SimModel contains the class SimReplication, which runs
#   a simulation replication and stores the simulation data
#   in its object attributes.
# InputOutputTools contains utility functions that act on
#   instances of SimReplication to load and export
#   simulation states and data.
# Tools_Optimization contains utility functions for optimization purposes.

import copy
from Engine_SimObjects import MultiTierPolicy, CDCTierPolicy
from Engine_DataObjects import DataPrepConfig, TimeSeriesManager, Calendar, City, TierInfo, Vaccine
from Engine_SimModel import SimReplication
import Tools_InputOutput
import Tools_Optimization_Utilities

# Import other Python packages
import numpy as np
import pandas as pd

import psutil

from pathlib import Path

import datetime as dt
import json

###############################################################################

import time

start_time = time.time()

process = psutil.Process()

transmission_df = pd.read_csv(DataPrepConfig.base_path / "instances" / "austin" / "transmission.csv")
setup_data = json.load(open(DataPrepConfig.base_path / "instances" / "austin" / "austin_setup.json"))

austin_calendar = Calendar("austin",
                           "calendar.csv",
                           "02/28/20",
                           945)

time_series_manager = TimeSeriesManager()
time_series_manager.create_fixed_time_series_from_monthdayyear_df("date",
                                                                  transmission_df,
                                                                  austin_calendar.simulation_datetimes)
time_series_manager.create_fixed_time_series_from_monthdayyear_intervals("school_closure",
                                                                         setup_data["school_closure"],
                                                                         austin_calendar.simulation_datetimes)

austin = City("austin",
              austin_calendar,
              "austin_setup.json",
              "variant.json",
              "austin_hospital_home_timeseries.csv",
              "variant_prevalence.csv")

tiers = TierInfo("austin", "tiers4.json")

vaccines = Vaccine(austin,
                   "austin",
                   "vaccines.json",
                   "booster_allocation_fixed.csv",
                   "vaccine_allocation_fixed.csv")

###############################################################################

# The following examples build on each other, so it is
#   recommended to study them in order.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Example A: Simulating a threshold policy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# In general, simulating a policy requires the steps
# (1) Create a MultiTierPolicy instance with desired thresholds.
# (2) Create a SimReplication instance with  aforementioned policy
#   and a random number seed -- this seed dictates the randomly sampled
#   epidemiological parameters for that replication as well as the
#   binomial random variable transitions between compartments in the
#   SEIR-type model.
# (3) Advance simulation time.

# Specify the 5 thresholds for a 5-tier policy
thresholds = (-1, 100, 200, 500, 1000)

# Create an instance of MultiTierPolicy using
#   austin, tiers (defined above)
#   thresholds (defined above)
#   "green" as the community_transmission toggle
# Prof Morton mentioned that setting community_transmission to "green"
#   was a government official request to stop certain "drop-offs"
#   in active tiers.
mtp = MultiTierPolicy(tiers, thresholds)

# Create an instance of SimReplication with seed 500.
rep = SimReplication(austin,
                     time_series_manager,
                     setup_data["epi_params"],
                     vaccines,
                     mtp,
                     100)

# Note that specifying a seed of -1 creates a simulation replication
#   with average values for the "random" epidemiological parameter
#   values and deterministic binomial transitions
#   (also taking average values).

# Advance simulation time until a desired end day.
# Currently, any non-negative integer between 0 and 963 (the length
#   of the user-specified "calendar.csv") works.
# Attributes in the SimReplication instance are updated in-place
#   to reflect the most current simulation state.
rep.simulate_time_period(945)

print(rep.compute_rsq())
print(np.sum(rep.ICU_history))

print(time.time() - start_time)

# print(process.memory_info().rss)

breakpoint()