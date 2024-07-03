###############################################################################

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

thresholds = (-1, 100, 200, 500)
# thresholds = (-1, 500, 500, 500)
mtp = MultiTierPolicy(tiers, thresholds)

austin_model.policy = mtp
austin_model.simulate_time_period(945)

print(austin_model.compute_rsq())
print(np.sum(austin_model.S))

breakpoint()