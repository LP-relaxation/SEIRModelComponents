###############################################################################

# Engine_SimModel.py
# This module contains the SimModel class. Each instance holds
#   a City instance, an EpiParams instance, a Vaccine instance,
#   VaccineGroup instance(s), and optionally a MultiTierPolicy instance.

###############################################################################

import numpy as np
import pandas as pd
import json
from Engine_DataObjects import EpiParams, DataPrepConfig, TimeSeriesManager, Calendar, City, Vaccine
from Engine_SimObjects import VaccineGroup

###############################################################################


class SimModelConstructor:

    def __init__(self,
                 city_name,
                 simulation_start_date_str,
                 max_simulation_length,
                 dict_filenames,
                 seed=100):

        self.transmission_df = pd.read_csv(DataPrepConfig.base_path / dict_filenames["historical_transmission_timeseries_csv"])
        self.setup_data = json.load(open(DataPrepConfig.base_path / dict_filenames["setup_json"]))
        self.calendar = Calendar(dict_filenames["calendar_csv"], simulation_start_date_str, max_simulation_length)

        self.time_series_manager = TimeSeriesManager()
        self.time_series_manager.create_fixed_time_series_from_monthdayyear_df("date",
                                                                               self.transmission_df,
                                                                               self.calendar.simulation_datetimes)
        self.time_series_manager.create_fixed_time_series_from_monthdayyear_intervals("school_closure",
                                                                                      self.setup_data["school_closure"],
                                                                                      self.calendar.simulation_datetimes)

        self.city = City(city_name,
                         self.calendar,
                         dict_filenames["setup_json"],
                         dict_filenames["variant_json"],
                         dict_filenames["historical_hospital_timeseries_csv"],
                         dict_filenames["variant_prevalence_csv"])

        self.vaccines = Vaccine(self.city,
                                dict_filenames["vaccine_info_json"],
                                dict_filenames["booster_allocation_csv"],
                                dict_filenames["vaccine_allocation_csv"])

        self.seed = seed

    def create_sim_model(self):
        return SimModel(self.city,
                        self.time_series_manager,
                        self.setup_data["epi_params"],
                        self.vaccines,
                        None,
                        self.seed)

class SimModel:
    def __init__(self,
                 instance,
                 time_series_manager,
                 epi_params,
                 vaccine,
                 policy,
                 rng_seed):
        """
        :param instance: [obj] instance of City class
        :param time_series_manager: [obj] instance of TimeSeriesManager class
        :param vaccine: [obj] instance of Vaccine class
        :param policy: [obj] instance of MultiTierPolicy
            class, or [None]
        :param rng_seed: [int] or [None] either a
            non-negative integer, -1, or None
        """

        self.instance = instance
        self.time_series_manager = time_series_manager
        self.vaccine = vaccine
        self.policy = policy
        self.rng_seed = rng_seed

        self.step_size = self.instance.step_size

        self.discrete_approx = self.build_discrete_approx_function(self.step_size)

        self.t_historical_data_end = len(self.instance.real_IH_history)
        self.fixed_kappa_end_date = 0

        # A is the number of age groups
        # L is the number of risk groups
        # Many data arrays in the simulation have dimension A x L
        A = self.instance.A
        L = self.instance.L

        # Important steps critical to initializing a replication
        # Initialize random number generator
        # Sample random parameters
        # Create new VaccineGroup instances
        self.init_rng()
        self.init_epi(epi_params, self.rng)
        self.init_vaccine_groups()

        self.vaccine.create_num_eligible_dict(self.instance.N,
                                              A*L,
                                              self.vaccine_groups,
                                              self.instance.cal.simulation_datetimes[-1])

        # Initialize data structures to track ICU, IH, flow_to_IH, flow_to_IY
        # These statistics or data we look at changes a lot over time
        # better to keep them in a list to modify.
        self.history_vars = ("ICU",
                             "IH",
                             "D",
                             "R",
                             "flow_to_IH",
                             "flow_to_IY",
                             "ICU_to_D",
                             "IY_to_D",
                             "flow_to_natural_immunity_S",
                             "flow_to_vaccine_induced_immunity_S",
                             "S")

        for attribute in self.history_vars:
            setattr(self, f"{attribute}_history", [])

        # The next t that is simulated (automatically gets updated after simulation)
        # This instance has simulated up to but not including time next_t
        self.next_t = 0

        # Tuples of variable names for organization purposes
        self.state_vars = ("S", "E", "IA", "IY", "PA", "PY", "R", "D", "IH", "ICU")
        self.tracking_vars = ("IY_to_IH", "IY_to_ICU", "IH_to_ICU", "flow_to_ICU", "flow_to_IH", "ICU_to_D",
                              "IY_to_D", "flow_to_IA", "flow_to_IY",
                              "flow_to_natural_immunity_S", "flow_to_vaccine_induced_immunity_S")

        self.total_imbalance = []

    def init_rng(self):
        """
        Assigns self.rng to a newly created random number generator
            initialized with seed self.rng_seed.
        If self.rng_seed is None (not specified) or -1, then self.rng
            is set to None, so no random number generator is created
            and the simulation will run deterministically.

        :return: [None]
        """

        if self.rng_seed is not None and self.rng_seed >= 0:
            self.rng = np.random.default_rng(self.rng_seed)
        else:
            self.rng = None

    def init_epi(self, epi_params, rng):
        """
        Assigns self.epi_rand to an instance of EpiParams that
            inherits some attribute values (primitives) from
            the "base" object self.instance.base_epi and
            also generates new values for other attributes.
        These new values come from randomly sampling
            parameters using the random number generator
            self.rng.
        If no random number generator is given, these
            randomly sampled parameters are set to the
            expected value from their distributions.
        After random sampling, some basic parameters
            are updated.

        :return: [None]
        """

        self.epi_rand = EpiParams(epi_params, rng)

        # Sample random parameters and
        #   do some basic updating based on the results
        #   of this sampling
        self.epi_rand.setup_base_params()

    def init_vaccine_groups(self):
        """
        Creates 4 vaccine groups:
            group 0 / "unvax": unvaccinated
            group 1 / "first_dose": partially vaccinated
            group 2 / "second_dose": fully vaccinated
            group 3 / "waned": waning efficacy

        We assume there is one type of vaccine with 2 doses.
        After 1 dose, individuals move from group 0 to 1.
        After 2 doses, individuals move from group 1 to group 2.
        After efficacy wanes, individuals move from group 2 to group 3.
        After booster shot, individuals move from group 3 to group 2.
                 - one type of vaccine with two-doses

        :return: [None]
        """

        N = self.instance.N
        I0 = self.instance.I0
        A = self.instance.A
        L = self.instance.L

        self.vaccine_groups = []

        self.vaccine_groups.append(VaccineGroup("unvax", 0, 0, 0, N, I0, A, L, self.step_size))
        for key in self.vaccine.beta_reduct:
            self.vaccine_groups.append(
                VaccineGroup(
                    key,
                    self.vaccine.beta_reduct[key],
                    self.vaccine.tau_reduct[key],
                    self.vaccine.pi_reduct[key],
                    N, I0, A, L, self.step_size
                )
            )
        self.vaccine_groups = tuple(self.vaccine_groups)

    def compute_cost(self):
        """
        If a policy is attached to this replication, return the
            cumulative cost of its enforced tiers (from
            the end of the historical data time period to the
            current time of the simulation).
        If no policy is attached to this replication, return
            None.

        :return: [float] or [None] cumulative cost of the
            attached policy's enforced tiers (returns None
            if there is no attached policy)
        """

        if self.policy:
            return sum(
                self.policy.tiers[i]["daily_cost"]
                for i in self.policy.tier_history
                if i is not None
            )
        else:
            return None

    def compute_feasibility(self):
        """
        If a policy is attached to this replication, return
            True/False if the policy is estimated to be
            feasible (from the end of the historical data time period
            to the current time of the simulation).
        If no policy is attached to this replication or the
            current time of the simulation is still within
            the historical data time period, return None.

        :return: [Boolean] or [None] corresponding to whether
            or not the policy is estimated to be feasible
        """

        if self.policy is None:
            return None

        # Check whether ICU capacity has been violated
        if np.any(
                np.array(self.ICU_history).sum(axis=(1, 2))[self.fixed_kappa_end_date:self.next_t]
                > self.instance.dedicated_covid_icu
        ):
            return False
        else:
            return True

    def compute_rsq(self):
        """
        Return R-squared type statistic based on historical hospital
            data (see pg. 10 in Yang et al. 2021), comparing
            thus-far-simulated hospital numbers (starting from t = 0
            up to the current time of the simulation) to the
            historical data hospital numbers (over this same time
            interval).

        Note that this statistic is not exactly R-squared --
            and as a result it takes values outside of [-1, 1].

        :return: [float] current R-squared value
        """

        f_benchmark = self.instance.real_IH_history

        IH_sim = np.array(self.ICU_history) + np.array(self.IH_history)
        IH_sim = IH_sim.sum(axis=(2, 1))
        IH_sim = IH_sim[: self.t_historical_data_end]

        if self.next_t < self.t_historical_data_end:
            IH_sim = IH_sim[: self.next_t]
            f_benchmark = f_benchmark[: self.next_t]

        rsq = 1 - np.sum(((np.array(IH_sim) - np.array(f_benchmark)) ** 2)) / sum(
            (np.array(f_benchmark) - np.mean(np.array(f_benchmark))) ** 2
        )

        return rsq

    def build_discrete_approx_function(self, step_size):

        def discrete_approx(rate):
            return 1 - np.exp(-rate / step_size)

        return discrete_approx

    def simulate_time_period(self, time_end):

        """
        Advance the simulation model from time_start up to
            but not including time_end.

        Calls simulate_t as a subroutine for each t between
            time_start and self.next_t, the last point at which it
            left off.

        Note that if a simulation replication is being run at timepoints
        t > fixed_kappa_end_date, there must be a MultiTierPolicy attached.

        :param time_end: [int] nonnegative integer -- time t (number of days)
            to simulate up to.
        :return: [None]
        """

        # Begin where the simulation last left off
        time_start = self.next_t

        # Call simulate_t as subroutine from time_start to time_end
        for t in range(time_start, time_end):

            self.next_t += 1

            self.simulate_t(t)
            # print(t)

            A = self.instance.A
            L = self.instance.L

            # Clear attributes in self.state_vars + self.tracking_vars
            #   since these only hold the most recent values
            for attribute in self.state_vars + self.tracking_vars:
                setattr(self, attribute, np.zeros((A, L)))

            # Update attributes in self.state_vars + self.tracking_vars --
            #   their values are the sums of the same attributes
            #   across all vaccine groups
            for attribute in self.state_vars + self.tracking_vars:
                sum_across_vaccine_groups = 0
                for v_group in self.vaccine_groups:

                    # Note: add this to tests! But take this out of code
                    # assert (getattr(v_group, attribute) > -1e-2).all(), \
                    #     f"fPop negative value of {getattr(v_group, attribute)} " \
                    #     f"on compartment {v_group.v_name}.{attribute} at time {self.instance.cal.calendar[t]}, {t}"

                    sum_across_vaccine_groups += getattr(v_group, attribute)

                    # Note: add this to tests! But take this out of code
                    # assert (getattr(v_group, attribute) > -1e-2).all(), \
                    #     f"fPop negative value of {getattr(v_group, attribute)} " \
                    #     f"on compartment {v_group.v_name}.{attribute} at time " \
                    #     f"{self.instance.cal.calendar[t]}, {t}"

                setattr(self, attribute, sum_across_vaccine_groups)

            # if t >= 570:
            #     print(t, np.sum(self.vaccine_groups[-2].S), np.sum(self.S), np.sum(self.E), np.sum(self.IA), np.sum(self.IY), np.sum(self.R), np.sum(self.D), np.sum(self.PA), np.sum(self.PY), np.sum(self.IH), np.sum(self.ICU))

            # We are interested in the history, not just current values, of
            #   certain variables -- save these current values
            for attribute in self.history_vars:
                getattr(self, f"{attribute}_history").append(getattr(self, attribute))

            # Note: add this to tests! But take this out of code
            total_imbalance = np.sum(
                self.S
                + self.E
                + self.IA
                + self.IY
                + self.R
                + self.D
                + self.PA
                + self.PY
                + self.IH
                + self.ICU
            ) - np.sum(self.instance.N)

            self.total_imbalance.append(total_imbalance)

            # if t >= 570:
            #     print("Total imbalance " + str(total_imbalance))

            # assert (
            #         np.abs(total_imbalance) < 1e-2
            # ), f"fPop unbalanced"

    def simulate_t(self, t_date):

        """
        Advance the simulation 1 timepoint (day).

        Subroutine called in simulate_time_period

        :param t_date: [int] nonnegative integer corresponding to
            current timepoint to simulate.
        :return: [None]
        """

        # Get dimensions (number of age groups,
        #   number of risk groups,
        A = self.instance.A
        L = self.instance.L
        N = self.instance.N

        calendar = self.instance.cal.simulation_datetimes

        t = t_date

        epi = self.epi_rand

        if t <= self.fixed_kappa_end_date:
            # If the transmission reduction is fixed don't call the policy object.
            phi_t = epi.effective_phi(
                self.time_series_manager.dict_names["school_closure"][t],
                self.time_series_manager.dict_names["cocooning"][t],
                self.time_series_manager.dict_names["transmission_reduction"][t],
                N / N.sum(),
                self.instance.cal._day_type[t],
            )
        else:
            current_tier = self.policy.get_current_tier(
                t,
                N,
                self.flow_to_IH_history,
                self.flow_to_IY_history,
                self.instance.moving_avg_len)
            phi_t = epi.effective_phi(
                self.policy.tiers[current_tier]["school_closure"],
                self.policy.tiers[current_tier]["cocooning"],
                self.policy.tiers[current_tier]["transmission_reduction"],
                N / N.sum(),
                self.instance.cal._day_type[t],
            )

        if calendar[t] >= self.instance.variant_start:
            t_since_variant = (calendar[t] - self.instance.variant_start).days

            epi_params_under_variants = self.instance.variant_pool.get_epi_params_under_variants(t_since_variant)
            vaccine_params_under_variants = self.instance.variant_pool.get_vaccine_params_under_variants(t_since_variant)

            # Assume immune evasion starts with the variants.
            immune_evasion = self.instance.variant_pool.immune_evasion(epi.immune_evasion, calendar[t])
            for v_group in self.vaccine_groups:

                total_variant_prevalence = self.instance.variant_pool.get_total_variant_prevalence(t_since_variant)
                if v_group.v_name != 'unvax':
                    v_group.variant_update(vaccine_params_under_variants, total_variant_prevalence)

            for (epi_param_name, val) in epi_params_under_variants.items():
               setattr(epi, epi_param_name, val * getattr(epi, epi_param_name + "0"))

            epi.sigma_E = self.instance.variant_pool.get_sigma_E_under_variant(epi.sigma_E0, t_since_variant)

        else:
            immune_evasion = 0

        # Updates pIH, HICUR, etaICU based on timeseries specified in
        #   otherInfo (additional columns in transmission.csv file)
        # From group discussion, we will first create version of code
        #   assuming all time series are specified
        # Afterwards, we will add optionality -- provide a backup option
        #   for values of pIH, HICUR, etaICU if user does not specify
        #   timeseries for one or more of these parameters
        #   (this was the previous functionality of rd_rate, etc...
        #   but this has been removed from the setup .json file)
        epi.update_icu_all(t, self.time_series_manager)

        step_size = self.step_size
        binomial_transition = self.binomial_transition

        nu_ICU = epi.nu_ICU
        nu = epi.nu

        discrete_approx = self.discrete_approx

        rate_E = discrete_approx(epi.sigma_E)
        rate_IAR = np.full((A, L), discrete_approx(epi.gamma_IA))
        rate_PAIA = np.full((A, L), discrete_approx(epi.rho_A))
        rate_PYIY = np.full((A, L), discrete_approx(epi.rho_Y))
        rate_IH_to_ICU = discrete_approx(nu * epi.etaICU)
        rate_IH_to_R = discrete_approx((1 - nu) * epi.gamma_IH)
        rate_ICU_to_D = discrete_approx(nu_ICU * epi.mu_ICU)
        rate_ICUR = discrete_approx((1 - nu_ICU) * epi.gamma_ICU)
        rate_immune = discrete_approx(immune_evasion)

        for _t in range(step_size):
            # Dynamics for dS

            dSprob_sum = np.zeros((5, 2))

            for v_group in self.vaccine_groups:

                # Compute the "baseline" probability of transitioning out of S
                # See [1m] in Arslan et al. (2023) -- $dS_{t,\omega}^{a,r,v}$
                #   is the same for all vaccine groups v *except* in
                #   transmission parameter $\beta^v$ that can be pulled out of
                #   the summands -- so we compute a "baseline" quantity and then
                #   adjust based on different transmission parameters for
                #   corresponding vaccine groups

                # epi.omega_PY and epi.omega_PA are 5x0

                # According to Arslan et al. (2023):
                # epi.omega_IA is infectiousness of individuals in IA
                #   relative to IY
                # epi.omega_IY is infectiousness of individuals in IY
                #   relative to IY (is equal to scalar 1)
                # Coefficients are correct -- epi.omega_PY = epi.omega_IY * epi.omega_P
                #   and epi.omega_IA = epi.omega_IA * epi.omega_P (see DataObjects)
                weighted_sum_infectious_groups = (
                        np.matmul(np.diag(epi.omega_PY), v_group._PY[_t, :, :])
                        + np.matmul(np.diag(epi.omega_PA), v_group._PA[_t, :, :])
                        + epi.omega_IA * v_group._IA[_t, :, :]
                        + epi.omega_IY * v_group._IY[_t, :, :]
                )

                # 5x1 array of total number in each age group
                #   (sum over risk groups)
                sum_individuals_count = np.sum(N, axis=1)[np.newaxis].T

                # phi_t is (5, 2, 5, 2) -- how a given age and risk group
                #   interacts with another age and risk group (5x2 by 5x2)
                # phi_t[i][0] == phi_t[i][1] (same 5x2 array)
                summand = np.divide(
                    np.multiply(epi.beta * phi_t / step_size, weighted_sum_infectious_groups), sum_individuals_count
                )

                # dSprob is 5x2
                # Summing over axis=(2,3) because summing over
                #   all age and risk groups a', r' --
                #   interested in how age and risk group a, r
                #   interacts with all other groups a', r' (including
                #   own group)
                dSprob = np.sum(summand, axis=(2, 3))
                dSprob_sum += dSprob

            for v_group in self.vaccine_groups:

                # Shortcuts for attribute access to speed up simulation time
                # Do not use shortcut if updating an attribute value --
                #   only use if want to quickly refer to attribute value
                # Shortcuts for all variables in v_group.state_vars
                #   defined in a chunk here, but shortcuts for
                #   some (not all) variables in v_group.tracking_vars
                #   defined after they are updated in the code
                # Not all variables in v_group.tracking_vars have
                #   shortcuts because many are only used once or twice
                S = v_group._S[_t]
                E = v_group._E[_t]
                IA = v_group._IA[_t]
                IY = v_group._IY[_t]
                PA = v_group._PA[_t]
                PY = v_group._PY[_t]
                R = v_group._R[_t]
                D = v_group._D[_t]
                IH = v_group._IH[_t]
                ICU = v_group._ICU[_t]

                # self.state_vars = ("S", "E", "IA", "IY", "PA", "PY", "R", "D", "IH", "ICU")

                _dSE = binomial_transition(S, (1 - v_group.v_beta_reduct) * dSprob_sum)

                # If there is immune evasion, there will be two outgoing arcs from S_vax.
                # Infected people move to E compartment.
                # People with waned immunity will go the S_waned compartment.
                # _dS: total rate for leaving S compartment.
                # _dSE: adjusted rate for entering E compartment.
                # _dSWaned: adjusted rate for entering S_waned (self.vaccine_groups[3]._S) compartment.
                if v_group.v_name == "second_dose":
                    _dSWaned = binomial_transition(S, rate_immune)
                else:
                    _dSWaned = 0

                _dS = _dSWaned + _dSE

                pi = epi.pi
                v_pi_reduct = v_group.v_pi_reduct
                gamma_IY = epi.gamma_IY
                alpha_IY_to_D = epi.alpha_IY_to_D
                pIH = epi.pIH
                Eta = epi.Eta

                rate_IYR = discrete_approx(np.array((1 - pi * (1 - v_pi_reduct)) * gamma_IY * (1 - alpha_IY_to_D)))
                rate_IY_to_D = discrete_approx(np.array((1 - pi * (1 - v_pi_reduct)) * gamma_IY * alpha_IY_to_D))
                rate_IYH = discrete_approx(np.array(pi * (1 - v_pi_reduct) * pIH) * np.expand_dims(Eta, axis=1))
                rate_IY_to_ICU = discrete_approx(np.array(pi * (1 - v_pi_reduct) * (1-pIH)) * np.expand_dims(Eta, axis=1))

                E_out = binomial_transition(E, rate_E)
                immune_escape_R = binomial_transition(R, rate_immune)
                EPY = binomial_transition(E_out, epi.tau * (1 - v_group.v_tau_reduct))
                PYIY = binomial_transition(PY, rate_PYIY)
                PAIA = binomial_transition(PA, rate_PAIA)
                IAR = binomial_transition(IA, rate_IAR)
                IYR = binomial_transition(IY, rate_IYR)
                IY_to_D = binomial_transition(IY - IYR, rate_IY_to_D)
                IY_to_IH = binomial_transition(IY - IYR - IY_to_D, rate_IYH)
                IY_to_ICU = binomial_transition(IY - IYR - IY_to_D - IY_to_IH, rate_IY_to_ICU)
                IH_to_R = binomial_transition(IH, rate_IH_to_R)
                IH_to_ICU = binomial_transition(IH - IH_to_R, rate_IH_to_ICU)
                ICUR = binomial_transition(ICU, rate_ICUR)
                ICU_to_D = binomial_transition(ICU - ICUR, rate_ICU_to_D)

                v_group._E[_t + 1] = E + _dSE - E_out

                self.vaccine_groups[3]._S[_t + 1] += _dSWaned + immune_escape_R

                # if np.sum(S) != 0:
                #     print(t, v_group.v_name, np.sum(v_group._S[_t+1]), np.sum(S), np.sum(_dS))
                v_group._S[_t + 1] += S - _dS

                v_group._PY[_t + 1] = PY + EPY - PYIY
                v_group._PA[_t + 1] = PA + E_out - EPY - PAIA
                v_group._IA[_t + 1] = IA + PAIA - IAR
                v_group._IY[_t + 1] = (IY + PYIY - IYR - IY_to_D - IY_to_IH - IY_to_ICU)
                v_group._IH[_t + 1] = (IH + IY_to_IH - IH_to_R - IH_to_ICU)
                v_group._ICU[_t + 1] = (ICU + IH_to_ICU - ICU_to_D - ICUR + IY_to_ICU)
                v_group._R[_t + 1] = (R + IH_to_R + IYR + IAR + ICUR - immune_escape_R)
                v_group._D[_t + 1] = D + ICU_to_D + IY_to_D

                v_group._IH_to_ICU[_t] = IH_to_ICU
                v_group._flow_to_ICU[_t] = IY_to_ICU + IH_to_ICU
                v_group._flow_to_IH[_t] = IY_to_ICU + IY_to_IH
                v_group._ICU_to_D[_t] = ICU_to_D
                v_group._IY_to_D[_t] = IY_to_D
                v_group._flow_to_IA[_t] = PAIA
                v_group._flow_to_IY[_t] = PYIY

                v_group._flow_to_vaccine_induced_immunity_S[_t] = _dSWaned
                v_group._flow_to_natural_immunity_S[_t] = immune_escape_R

        for v_group in self.vaccine_groups:
            for attribute in self.state_vars:
                setattr(v_group, attribute, getattr(v_group, "_" + attribute)[step_size].copy())

            for attribute in self.tracking_vars:
                setattr(v_group, attribute, getattr(v_group, "_" + attribute).sum(axis=0))

        if t >= self.vaccine.vaccine_start_time:
            self.vaccine_schedule(t, rate_immune)

        for v_group in self.vaccine_groups:
            for attribute in self.state_vars:
                setattr(v_group, "_" + attribute, np.zeros((step_size + 1, A, L)))
                vars(v_group)["_" + attribute][0] = vars(v_group)[attribute]

            for attribute in self.tracking_vars:
                setattr(v_group, "_" + attribute, np.zeros((step_size, A, L)))

    def vaccine_schedule(self, t, rate_immune):
        """
        Mechanically move people between compartments for daily vaccination at the end of a day. We only move people
        between the susceptible compartments but assume that people may receive vaccine while they are already infected
        of recovered (in that case we assume that the vaccine is wasted). For vaccine amount of X, we adjust
        it by X * (S_v/N_v) where S_v is the total population in susceptible for vaccine status v and N_v is
        the total eligible population (see Vaccine.get_num_eligible for more detail) for vaccination in
        vaccine status v.
            The vaccination assumption is slightly different for booster dose. People don't mechanically move to the
        waned compartment, we draw binomial samples with a certain rate, but they get booster mechanically. That's why
        the definition of total eligible population is different. We adjust the vaccine amount X as
        X * (S_waned/(N_waned + N_fully_vax)) where N_waned and N_fully_wax is the total population in waned and
        fully vax compartments (see VaccineGroup.get_total_population). We estimate the probability of being waned and
        susceptible with S_waned/(N_waned + N_fully_vax).
        :param t_date: the current day simulated.
        :param rate_immune: immune evasion rate. Adjust the vaccine amount if there is immune evasion.
        """
        A = self.instance.A
        L = self.instance.L
        N = self.instance.N

        current_datetime = self.instance.cal.simulation_datetimes[t]

        S_temp = {}
        N_temp = {}

        S_before = np.zeros((5, 2))

        for v_group in self.vaccine_groups:
            S_before += v_group.S
            S_temp[v_group.v_name] = v_group.S
            N_temp[v_group.v_name] = v_group.get_total_population(A * L)

        for v_group in self.vaccine_groups:

            out_sum = np.zeros((A, L))

            for vaccine_type in v_group.v_out:
                if current_datetime in self.vaccine.vaccine_allocation[vaccine_type]:
                    S_out = np.reshape(
                        self.vaccine.vaccine_allocation[vaccine_type][current_datetime],
                        (A * L, 1),
                    )

                    if v_group.v_name == "waned":
                        num_eligible = N_temp["waned"] + N_temp["second_dose"]
                    else:
                        num_eligible = self.vaccine.num_eligible_dict[v_group.v_name][current_datetime]

                    ratio_S_N = np.array(
                        [
                            0 if num_eligible[i] == 0 else float(S_out[i] / num_eligible[i])
                            for i in range(len(num_eligible))
                        ]
                    ).reshape((A, L))

                    out_sum += (ratio_S_N * S_temp[v_group.v_name]).astype(int)

            in_sum = np.zeros((A, L))

            for vaccine_type in v_group.v_in:

                for v_g in self.vaccine_groups:
                    if ((v_g.v_name == "waned" and vaccine_type == "booster") or
                            (v_g.v_name == "first_dose" and vaccine_type == "second_dose") or
                            (v_g.v_name == "unvax" and vaccine_type == "first_dose")):
                        previous_vaccine_group = v_g

                if current_datetime in self.vaccine.vaccine_allocation[vaccine_type]:
                    S_in = np.reshape(
                        self.vaccine.vaccine_allocation[vaccine_type][current_datetime],
                        (A * L, 1),
                    )

                    if previous_vaccine_group.v_name == "waned":
                        num_eligible = N_temp["waned"] + N_temp["second_dose"]
                    else:

                        num_eligible = self.vaccine.num_eligible_dict[previous_vaccine_group.v_name][current_datetime]

                    ratio_S_N = np.array(
                        [
                            0 if num_eligible[i] == 0 else float(S_in[i] / num_eligible[i])
                            for i in range(len(num_eligible))
                        ]
                    ).reshape((A, L))

                    in_sum += (ratio_S_N * S_temp[previous_vaccine_group.v_name]).astype(int)

            v_group.S = v_group.S + (np.array(in_sum - out_sum))

            # if t == 577 and v_group.v_name == "second_dose":
            #     breakpoint()

        S_after = np.zeros((5, 2))

        for v_group in self.vaccine_groups:
            S_after += v_group.S

        imbalance = np.abs(np.sum(S_before - S_after, axis=(0, 1)))

        # # Note: add this to tests! But take this out of code
        assert (imbalance < 1e-2).all(), (
            f"fPop inbalance in vaccine flow in between compartment S "
            f"{imbalance} at time {t}"
        )

    def reset(self):

        '''
        In-place "resets" the simulation by clearing simulation data
            and reverting the time simulated to 0.
        Does not reset the random number generator, so
            simulating a replication that has been reset
            pulls new random numbers from where the random
            number generator last left off.
        Does not reset or resample the previously randomly sampled
            epidemiological parameters either.

        :return: [None]
        '''

        self.init_vaccine_groups()

        for attribute in self.history_vars:
            setattr(self, f"{attribute}_history", [])

        self.next_t = 0

    def binomial_transition(self, n, p):

        '''
        Either returns mean value of binomial distribution
            with parameters n, p or samples from that
            distribution, depending on whether self.rng is
            specified (depending on if simulation is run
            deterministically or not).

        :param n: [float] nonnegative value -- can be non-integer
            if running simulation deterministically, but must
            be integer to run simulation stochastically since
            it is parameter of binomial distribution
        :param p: [float] value in [0,1] corresponding to
            probability parameter in binomial distribution
        :return: [int] nonnegative integer that is a realization
            of a binomial random variable
        '''

        if self.rng is None:
            return n * p
        else:
            return self.rng.binomial(np.round(n).astype(int), p)
