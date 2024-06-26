###############################################################################

# Engine_SimObjects.py

###############################################################################

import numpy as np

###############################################################################
# Common simple functions:


def find_tier(thresholds, stat):
    """
    Calculate the new tier according to the tier statistics.
    :param thresholds: the tier thresholds.
    :param stat: the critical statistics that would determine the next tier.
    :return: the new tier.
    """
    lb_threshold = 0
    for i, lt in enumerate(thresholds):
        if stat >= lt:
            lb_threshold = i
        else:
            break

    return lb_threshold


###############################################################################
# Modules:

class CDCTierPolicy:
    """
    CDC's community levels.
    CDC system includes three tiers. Green and orange stages are deprecated but maintained
    for code consistency with our system.
    CDC system includes three indicators;
        1. Case counts (new COVID-19 Cases Per 100,000 people in the past 7 days.),
        2. Hospital admissions (new COVID-19 admissions per 100,000 population (7-day total)),
        3. Percent hospital beds (percent of staffed inpatient beds occupied by COVID-19 patients (7-day average)).

        Depending on the case counts thresholds, the hospital admissions and percent hospital beds thresholds
    changes. I think of this as follows, when there is a surge of cases the other two thresholds are stricter
    but when there is no surge of cases the other two thresholds are more relax.
        The history for the case counts is written as self.surge_history to indicate which set of thresholds are active
    for hospital admissions and percent hospital beds.
        The new tier will be stricter of what hospital admission and percent hospital beds thresholds are indicating.
    """

    def __init__(self, instance,
                 tiers,
                 case_threshold,
                 hosp_adm_thresholds,
                 staffed_bed_thresholds,
                 specified_total_hosp_beds=None,
                 percentage_cases=0.4):
        """
        :param instance:
        :param tiers: (list of dict): a list of the tiers characterized by a dictionary
                with the following entries:
                    {
                        "transmission_reduction": float [0,1)
                        "cocooning": float [0,1)
                        "school_closure": int {0,1}
                    }
        :param case_threshold: (Surge threshold).
        :param hosp_adm_thresholds: (dict of dict) thresholds
                   { non_surge : thresholds level when case counts is below the case threshold
                    surge : thresholds level when case counts is above the case threshold
                   }
        :param staffed_bed_thresholds: (dict of dict) similar entries as the hosp_adm_thresholds.
        :param specified_total_hosp_beds [None] or [int]: if None, will use total_hosp_beds
            from instance (from setup json file) as total hospital capacity. Otherwise,
            will use specified integer instead.
        :param percentage_cases: the CDC system uses total case counts as an indicators. However, we don't have a direct
        interpretation of case counts in the model. We estimate the real total case count as some percentage of people
        entering symptomatic compartment (ToIY). We use percentage_case to adjust ToIY.
        """
        self._instance = instance
        self.tiers = tiers.tier
        self.case_threshold = case_threshold
        self.hosp_adm_thresholds = hosp_adm_thresholds
        self.staffed_bed_thresholds = staffed_bed_thresholds
        self.specified_total_hosp_beds = specified_total_hosp_beds
        self.percentage_cases = percentage_cases
        self.tier_history = None
        self.surge_history = None
        self.active_indicator_history = []

    def reset(self):
        self.tier_history = None
        self.surge_history = None
        self.active_indicator_history = []

    def __repr__(self):
        return f"CDC_{self.case_threshold}_{self.hosp_adm_thresholds['non_surge'][0]}_{self.staffed_bed_thresholds['non_surge'][0]}_{self.percentage_cases}"

    def __call__(self, t, ToIHT, IH, ToIY, ICU):
        N = self._instance.N

        if self.tier_history is None:
            self.tier_history = [None for i in range(t)]
            self.surge_history = [None for i in range(t)]
            self.active_indicator_history = [None for i in range(t)]
        if len(self.tier_history) > t:
            return

        ToIHT = np.array(ToIHT)
        IH = np.array(IH)
        ToIY = np.array(ToIY)
        ICU = np.array(ICU)

        # Compute daily admissions moving sum
        moving_avg_start = np.maximum(0, t - self._instance.moving_avg_len)
        hos_adm_total = ToIHT.sum((1, 2))
        hosp_adm_sum = 100000 * hos_adm_total[moving_avg_start:].sum() / N.sum((0, 1))

        # Compute 7-day total new cases:
        N = self._instance.N
        ToIY_total = ToIY.sum((1, 2))
        ToIY_total = ToIY_total[moving_avg_start:].sum() * 100000 / np.sum(N, axis=(0, 1))

        # Compute 7-day average percent of COVID beds:
        IH_total = IH.sum((1, 2)) + ICU.sum((1, 2))
        if self.specified_total_hosp_beds is None:
            IH_avg = IH_total[moving_avg_start:].mean() / self._instance.total_hosp_beds
        else:
            IH_avg = IH_total[moving_avg_start:].mean() / self.specified_total_hosp_beds

        current_tier = self.tier_history[t - 1]

        # Decide on the active hospital admission and staffed bed thresholds depending on the estimated
        # case count level:
        if ToIY_total * self.percentage_cases < self.case_threshold:
            hosp_adm_thresholds = self.hosp_adm_thresholds["non_surge"]
            staffed_bed_thresholds = self.staffed_bed_thresholds["non_surge"]
            surge_state = 0
        else:
            hosp_adm_thresholds = self.hosp_adm_thresholds["surge"]
            staffed_bed_thresholds = self.staffed_bed_thresholds["surge"]
            surge_state = 1

        # find hosp admission new tier:
        hosp_adm_tier = find_tier(hosp_adm_thresholds, hosp_adm_sum)

        # find staffed bed new tier:
        staffed_bed_tier = find_tier(staffed_bed_thresholds, IH_avg)

        # choose the stricter tier among tiers the two indicators suggesting:
        new_tier = max(hosp_adm_tier, staffed_bed_tier)
        # keep track of the active indicator for indicator statistics:
        if hosp_adm_tier > staffed_bed_tier:
            active_indicator = 0
        elif hosp_adm_tier < staffed_bed_tier:
            active_indicator = 1
        else:
            active_indicator = 2

        if current_tier != new_tier:  # bump to the next tier
            t_end = t + self.tiers[new_tier]["min_enforcing_time"]
        else:  # stay in same tier for one more time period
            new_tier = current_tier
            t_end = t + 1

        self.tier_history += [new_tier for i in range(t_end - t)]
        self.surge_history += [surge_state for i in range(t_end - t)]
        self.active_indicator_history += [active_indicator for i in range(t_end - t)]


class MultiTierPolicy:
    """
    A multi-tier policy allows for multiple tiers of lock-downs.
    Attrs:
        tiers (list of dict): a list of the tiers characterized by a dictionary
            with the following entries:
                {
                    "transmission_reduction": float [0,1)
                    "cocooning": float [0,1)
                    "school_closure": int {0,1}
                }

        lockdown_thresholds (list of list): a list with the thresholds for every
            tier. The list must have n-1 elements if there are n tiers. Each threshold
            is a list of values for evert time step of simulation.
    """

    def __init__(self, tiers, lockdown_thresholds):

        self.tiers = tiers.tier
        self.lockdown_thresholds = lockdown_thresholds

        self.tier_history = []
        self.days_since_tier_change = 0
        self.min_required_days_in_tier = tiers.min_required_days_in_tier
        self.days_spent_in_current_tier = 0

    def reset(self):

        self.tier_history = []
        self.days_since_tier_change = 0
        self.days_spent_in_current_tier = 0

    def __repr__(self):
        return str(self.lockdown_thresholds)

    def get_current_tier(self, t, N, ToIHT, ToIY, moving_avg_len):

        ToIHT = np.array(ToIHT)
        ToIY = np.array(ToIY)

        # Compute daily admissions moving average
        moving_avg_start = np.maximum(0, t - moving_avg_len)

        if len(ToIHT) > 0:
            critical_stat_avg = ToIHT.sum((1,2))[moving_avg_start:].mean()
        else:
            critical_stat_avg = 0

        # Compute new cases per 100k:
        if len(ToIY) > 0:
            ToIY_avg = ToIY.sum((1, 2))[moving_avg_start:].sum() * 1e5 / np.sum(N)
        else:
            ToIY_avg = 0

        # find new tier
        new_tier = find_tier(self.lockdown_thresholds, critical_stat_avg)

        # This code corresponds to the deprecated community_transmission attribute
        #   when it was set to "green" (default)
        # This was added after input from public health officials
        #   to stop certain "drop-offs" in active tiers.
        if new_tier == 0:
            if ToIY_avg > 5:
                if ToIY_avg < 10:
                    new_tier = 1
                else:
                    new_tier = 2

        if len(self.tier_history) > 0:
            previous_tier = self.tier_history[-1]
        else:
            previous_tier = None

        if previous_tier == None or (new_tier != previous_tier and self.days_since_tier_change >= self.min_required_days_in_tier):
            self.tier_history.append(new_tier)
            self.days_since_tier_change = 0
        else:
            self.tier_history.append(previous_tier)
            self.days_since_tier_change += 1

        return self.tier_history[-1]


class VaccineGroup:
    def __init__(
            self,
            v_name,
            v_beta_reduct,
            v_tau_reduct,
            v_pi_reduct,
            N, I0, A, L, step_size):

        """
        Define each vaccine status as a group. Define each set of compartments for vaccine group.
        """

        self.v_beta_reduct = v_beta_reduct
        self.v_tau_reduct = v_tau_reduct
        self.v_pi_reduct = v_pi_reduct
        self.v_name = v_name

        if self.v_name == "unvax":
            self.v_in = ()
            self.v_out = ("first_dose",)

        elif self.v_name == "first_dose":
            self.v_in = ("first_dose",)
            self.v_out = ("second_dose",)

        elif self.v_name == "second_dose":
            self.v_in = ("second_dose", "booster")
            self.v_out = ()

        else:
            self.v_in = ()
            self.v_out = ("booster",)

        self.N = N
        self.I0 = I0

        self.state_vars = ("S", "E", "IA", "IY", "PA", "PY", "R", "D", "IH", "ICU")
        self.tracking_vars = (
            "IY_to_IH",
            "IY_to_ICU",
            "IH_to_ICU",
            "flow_to_ICU",
            "flow_to_IH",
            "ICU_to_D",
            "IY_to_D",
            "flow_to_IA",
            "flow_to_IY",
            "flow_to_natural_immunity_S",
            "flow_to_vaccine_induced_immunity_S")

        for attribute in self.state_vars:
            setattr(self, attribute, np.zeros((A, L)))
            setattr(self, "_" + attribute, np.zeros((step_size + 1, A, L)))

        for attribute in self.tracking_vars:
            setattr(self, attribute, np.zeros((A, L)))
            setattr(self, "_" + attribute, np.zeros((step_size, A, L)))

        if self.v_name == "unvax":
            # Initial Conditions (assumed)
            self.PY = self.I0
            self.R = 0
            self.S = self.N - self.PY - self.IY

        for attribute in self.state_vars:
            vars(self)["_" + attribute][0] = getattr(self, attribute)

    def variant_update(self, params, prev):
        """
        Update efficacy according to variant of concern efficacy
        """
        self.v_beta_reduct = self.v_beta_reduct * (1 - prev) + params[
            ('v_beta_reduct', self.v_name)]  # efficacy against infection.
        self.v_tau_reduct = self.v_tau_reduct * (1 - prev) + params[
            ('v_tau_reduct', self.v_name)]  # efficacy against symptomatic infection.

    def get_total_population(self, total_risk_groups):
        """
        :param total_risk_groups: total number of compartments for age-risk groups.
        :return: the total population in a certain vaccine group (S+E+IY+PY+..).
        """
        N = 0
        for attribute in self.state_vars:
            N += getattr(self, attribute)
        N = N.reshape((total_risk_groups, 1))
        return N
