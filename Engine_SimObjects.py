###############################################################################

# Engine_SimObjects.py

###############################################################################

import numpy as np

###############################################################################

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

        if previous_tier == None or \
                (new_tier != previous_tier and self.days_since_tier_change >= self.min_required_days_in_tier):
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
