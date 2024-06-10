###############################################################################

# Engine_SimModel.py
# This module contains the SimReplication class. Each instance holds
#   a City instance, an EpiParams instance, a Vaccine instance,
#   VaccineGroup instance(s), and optionally a MultiTierPolicy instance.

###############################################################################

import numpy as np
from Engine_DataObjects import EpiParams
from Engine_SimObjects import VaccineGroup
import copy

###############################################################################

def discrete_approx(rate, timestep):
    return 1 - np.exp(-rate / timestep)


class SimReplication:
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

        # Save arguments as attributes
        self.instance = instance
        self.time_series_manager = time_series_manager
        self.vaccine = vaccine
        self.policy = policy
        self.rng_seed = rng_seed

        self.step_size = self.instance.step_size
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

        # Initialize data structures to track ICU, IH, ToIHT, ToIY
        # These statistics or data we look at changes a lot over time
        # better to keep them in a list to modify.
        self.history_vars = ("ICU",
                             "IH",
                             "D",
                             "R",
                             "ToIHT",
                             "ToIY",
                             "ToICUD",
                             "ToIYD",
                             "ToRS",
                             "ToSS",
                             "S")

        # Keep track of the total number of immune evasion:
        self.ToRS_immune = []  # waned natural immunity
        self.ToSS_immune = []  # waned vaccine induced immunity

        for attribute in self.history_vars:
            setattr(self, f"{attribute}_history", [])

        # The next t that is simulated (automatically gets updated after simulation)
        # This instance has simulated up to but not including time next_t
        self.next_t = 0

        # Tuples of variable names for organization purposes
        self.state_vars = ("S", "E", "IA", "IY", "PA", "PY", "R", "D", "IH", "ICU")
        self.tracking_vars = (
            "IYIH",
            "IYICU",
            "IHICU",
            "ToICU",
            "ToIHT",
            "ToICUD",
            "ToIYD",
            "ToIA",
            "ToIY",
            "ToRS",
            "ToSS"
        )

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

        self.base_epi = EpiParams(epi_params, rng)

        # Create a deep copy of the "base" EpiParams instance
        #   to inherit some attribute values (primitives)
        epi_rand = copy.deepcopy(self.base_epi)

        # On this copy, sample random parameters and
        #   do some basic updating based on the results
        #   of this sampling
        epi_rand.setup_base_params()

        # Assign self.epi_rand to this copy
        self.epi_rand = epi_rand

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
        step_size = self.instance.step_size

        self.vaccine_groups = []

        self.vaccine_groups.append(VaccineGroup("unvax", 0, 0, 0, N, I0, A, L, step_size))
        for key in self.vaccine.beta_reduct:
            self.vaccine_groups.append(
                VaccineGroup(
                    key,
                    self.vaccine.beta_reduct[key],
                    self.vaccine.tau_reduct[key],
                    self.vaccine.pi_reduct[key],
                    N, I0, A, L, step_size
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

            # We are interested in the history, not just current values, of
            #   certain variables -- save these current values
            for attribute in self.history_vars:
                getattr(self, f"{attribute}_history").append(getattr(self, attribute))

            # Note: add this to tests! But take this out of code
            # total_imbalance = np.sum(
            #     self.S
            #     + self.E
            #     + self.IA
            #     + self.IY
            #     + self.R
            #     + self.D
            #     + self.PA
            #     + self.PY
            #     + self.IH
            #     + self.ICU
            # ) - np.sum(self.instance.N)
            #
            # assert (
            #         np.abs(total_imbalance) < 1e-2
            # ), f"fPop unbalanced {total_imbalance} at time {self.instance.cal.calendar[t]}, {t}"

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

        epi = copy.deepcopy(self.epi_rand)

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
            self.policy(
                t,
                self.ToIHT_history,
                self.IH_history,
                self.ToIY_history,
                self.ICU_history,
            )
            current_tier = self.policy.tier_history[t]
            phi_t = epi.effective_phi(
                self.policy.tiers[current_tier]["school_closure"],
                self.policy.tiers[current_tier]["cocooning"],
                self.policy.tiers[current_tier]["transmission_reduction"],
                N / N.sum(),
                self.instance.cal._day_type[t],
            )

        if calendar[t] >= self.instance.variant_start:
            days_since_variant_start = (calendar[t] - self.instance.variant_start).days
            new_epi_params_coef, new_vax_params, var_prev = self.instance.variant_pool.update_params_coef(
                days_since_variant_start, epi.sigma_E)
            # Assume immune evasion starts with the variants.
            immune_evasion = self.instance.variant_pool.immune_evasion(epi.immune_evasion, calendar[t])
            for v_group in self.vaccine_groups:
                if v_group.v_name != 'unvax':
                    v_group.variant_update(new_vax_params, var_prev)
            epi.variant_update_param(new_epi_params_coef)
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
        get_binomial_transition_quantity = self.get_binomial_transition_quantity

        nu_ICU = epi.nu_ICU
        nu = epi.nu

        rate_E = discrete_approx(epi.sigma_E, step_size)
        rate_IAR = np.full((A, L), discrete_approx(epi.gamma_IA, step_size))
        rate_PAIA = np.full((A, L), discrete_approx(epi.rho_A, step_size))
        rate_PYIY = np.full((A, L), discrete_approx(epi.rho_Y, step_size))
        rate_IHICU = discrete_approx(nu * epi.etaICU, step_size)
        rate_IHR = discrete_approx((1 - nu) * epi.gamma_IH, step_size)
        rate_ICUD = discrete_approx(nu_ICU * epi.mu_ICU, step_size)
        rate_ICUR = discrete_approx((1 - nu_ICU) * epi.gamma_ICU, step_size)
        rate_immune = discrete_approx(immune_evasion, step_size)

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

                dSprob_sum = dSprob_sum + dSprob

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
                _S_t = v_group._S[_t]
                _E_t = v_group._E[_t]
                _IA_t = v_group._IA[_t]
                _IY_t = v_group._IY[_t]
                _PA_t = v_group._PA[_t]
                _PY_t = v_group._PY[_t]
                _R_t = v_group._R[_t]
                _D_t = v_group._D[_t]
                _IH_t = v_group._IH[_t]
                _ICU_t = v_group._ICU[_t]

                # self.state_vars = ("S", "E", "IA", "IY", "PA", "PY", "R", "D", "IH", "ICU")

                if v_group.v_name in {"second_dose"}:
                    # If there is immune evasion, there will be two outgoing arcs from S_vax.
                    # Infected people move to E compartment.
                    # People with waned immunity will go the S_waned compartment.
                    # _dS: total rate for leaving S compartment.
                    # _dSE: adjusted rate for entering E compartment.
                    # _dSR: adjusted rate for entering S_waned (self.vaccine_groups[3]._S) compartment.

                    _dSE = get_binomial_transition_quantity(
                        _S_t,
                        (1 - v_group.v_beta_reduct) * dSprob_sum,
                    )
                    _dSR = get_binomial_transition_quantity(_S_t, rate_immune)
                    _dS = _dSR + _dSE

                    E_out = get_binomial_transition_quantity(_E_t, rate_E)
                    v_group._E[_t + 1] = _E_t + _dSE - E_out

                    self.vaccine_groups[3]._S[_t + 1] = (
                            self.vaccine_groups[3]._S[_t + 1] + _dSR
                    )
                    v_group._ToSS[_t] = _dSR
                else:
                    _dS = get_binomial_transition_quantity(
                        _S_t, (1 - v_group.v_beta_reduct) * dSprob_sum
                    )
                    # Dynamics for E
                    E_out = get_binomial_transition_quantity(_E_t, rate_E)
                    v_group._E[_t + 1] = _E_t + _dS - E_out

                    v_group._ToSS[_t] = 0

                immune_escape_R = get_binomial_transition_quantity(_R_t, rate_immune)
                self.vaccine_groups[3]._S[_t + 1] = self.vaccine_groups[3]._S[_t + 1] + immune_escape_R
                v_group._ToRS[_t] = immune_escape_R
                v_group._S[_t + 1] += _S_t - _dS

                # Dynamics for PY
                EPY = get_binomial_transition_quantity(
                    E_out, epi.tau * (1 - v_group.v_tau_reduct)
                )
                PYIY = get_binomial_transition_quantity(_PY_t, rate_PYIY)
                v_group._PY[_t + 1] = _PY_t + EPY - PYIY

                # Dynamics for PA
                EPA = E_out - EPY
                PAIA = get_binomial_transition_quantity(_PA_t, rate_PAIA)
                v_group._PA[_t + 1] = _PA_t + EPA - PAIA

                # Dynamics for IA
                IAR = get_binomial_transition_quantity(_IA_t, rate_IAR)
                v_group._IA[_t + 1] = _IA_t + PAIA - IAR

                # Dynamics for IY
                rate_IYR = discrete_approx(
                    np.array(
                        [
                            [
                                (1 - epi.pi[a, l] * (1 - v_group.v_pi_reduct)) * epi.gamma_IY * (1 - epi.alpha_IYD)
                                for l in range(L)
                            ]
                            for a in range(A)
                        ]
                    ),
                    step_size,
                )
                rate_IYD = discrete_approx(
                    np.array(
                        [
                            [(1 - epi.pi[a, l] * (1 - v_group.v_pi_reduct)) * epi.gamma_IY * epi.alpha_IYD for l in
                             range(L)]
                            for a in range(A)
                        ]
                    ),
                    step_size,
                )
                IYR = get_binomial_transition_quantity(_IY_t, rate_IYR)
                IYD = get_binomial_transition_quantity(_IY_t - IYR, rate_IYD)

                rate_IYH = discrete_approx(
                    np.array(
                        [
                            [(epi.pi[a, l]) * (1 - v_group.v_pi_reduct) * epi.Eta[a] * epi.pIH for l in range(L)]
                            for a in range(A)
                        ]
                    ),
                    step_size,
                )
                rate_IYICU = discrete_approx(
                    np.array(
                        [
                            [(epi.pi[a, l]) * (1 - v_group.v_pi_reduct) * epi.Eta[a] * (1 - epi.pIH) for l in range(L)]
                            for a in range(A)
                        ]
                    ),
                    step_size,
                )

                v_group._IYIH[_t] = get_binomial_transition_quantity(
                    _IY_t - IYR - IYD, rate_IYH
                )

                _IYIH_t = v_group._IYIH[_t]

                v_group._IYICU[_t] = get_binomial_transition_quantity(
                    _IY_t - IYR - IYD - _IYIH_t, rate_IYICU
                )

                _IYICU_t = v_group._IYICU[_t]

                v_group._IY[_t + 1] = (
                        _IY_t
                        + PYIY
                        - IYR
                        - IYD
                        - _IYIH_t
                        - _IYICU_t
                )

                # Dynamics for IH
                IHR = get_binomial_transition_quantity(_IH_t, rate_IHR)
                v_group._IHICU[_t] = get_binomial_transition_quantity(
                    _IH_t - IHR, rate_IHICU
                )

                _IHICU_t = v_group._IHICU[_t]

                v_group._IH[_t + 1] = (
                        _IH_t + _IYIH_t - IHR - _IHICU_t
                )

                # Dynamics for ICU
                ICUR = get_binomial_transition_quantity(_ICU_t, rate_ICUR)
                ICUD = get_binomial_transition_quantity(
                    _ICU_t - ICUR, rate_ICUD
                )
                v_group._ICU[_t + 1] = (
                        _ICU_t
                        + _IHICU_t
                        - ICUD
                        - ICUR
                        + _IYICU_t
                )
                v_group._ToICU[_t] = _IYICU_t + _IHICU_t
                v_group._ToIHT[_t] = _IYICU_t + _IYIH_t

                # Dynamics for R
                v_group._R[_t + 1] = (_R_t + IHR + IYR + IAR + ICUR - immune_escape_R)

                # Dynamics for D
                v_group._D[_t + 1] = _D_t + ICUD + IYD
                v_group._ToICUD[_t] = ICUD
                v_group._ToIYD[_t] = IYD
                v_group._ToIA[_t] = PAIA
                v_group._ToIY[_t] = PYIY

        for v_group in self.vaccine_groups:
            # End of the daily discretization
            for attribute in self.state_vars:
                setattr(
                    v_group,
                    attribute,
                    getattr(v_group, "_" + attribute)[step_size].copy(),
                )

            for attribute in self.tracking_vars:
                setattr(
                    v_group, attribute, getattr(v_group, "_" + attribute).sum(axis=0)
                )

        if t >= self.vaccine.vaccine_start_time:
            self.vaccine_schedule(t, rate_immune)

        for v_group in self.vaccine_groups:

            for attribute in self.state_vars:
                setattr(v_group, "_" + attribute, np.zeros((step_size + 1, A, L)))
                vars(v_group)["_" + attribute][0] = vars(v_group)[attribute]

            for attribute in self.tracking_vars:
                setattr(v_group, "_" + attribute, np.zeros((step_size, A, L)))

    def vaccine_schedule(self, t_date, rate_immune):
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

        calendar = self.instance.cal.simulation_datetimes
        t = t_date
        S_before = np.zeros((5, 2))
        S_temp = {}
        N_temp = {}

        for v_group in self.vaccine_groups:
            S_before += v_group.S
            S_temp[v_group.v_name] = v_group.S
            N_temp[v_group.v_name] = v_group.get_total_population(A * L)

        for v_group in self.vaccine_groups:
            out_sum = np.zeros((A, L))
            S_out = np.zeros((A * L, 1))
            N_out = np.zeros((A * L, 1))

            for vaccine_type in v_group.v_out:
                event = self.vaccine.event_lookup(vaccine_type, calendar[t])

                if event is not None:
                    S_out = np.reshape(
                        self.vaccine.vaccine_allocation[vaccine_type][event][
                            "assignment"
                        ],
                        (A * L, 1),
                    )

                    if v_group.v_name == "waned":
                        N_out = N_temp["waned"] + N_temp["second_dose"]
                    else:
                        N_out = self.vaccine.get_num_eligible(
                            N,
                            A * L,
                            v_group.v_name,
                            v_group.v_in,
                            v_group.v_out,
                            calendar[t],
                        )
                    ratio_S_N = np.array(
                        [
                            0 if N_out[i] == 0 else float(S_out[i] / N_out[i])
                            for i in range(len(N_out))
                        ]
                    ).reshape((A, L))

                    out_sum += (ratio_S_N * S_temp[v_group.v_name]).astype(int)

            in_sum = np.zeros((A, L))
            S_in = np.zeros((A * L, 1))
            N_in = np.zeros((A * L, 1))
            for vaccine_type in v_group.v_in:
                for v_g in self.vaccine_groups:
                    if (
                            v_g.v_name
                            == self.vaccine.vaccine_allocation[vaccine_type][0]["from"]
                    ):
                        v_temp = v_g

                event = self.vaccine.event_lookup(vaccine_type, calendar[t])

                if event is not None:
                    S_in = np.reshape(
                        self.vaccine.vaccine_allocation[vaccine_type][event][
                            "assignment"
                        ],
                        (A * L, 1),
                    )

                    if v_temp.v_name == "waned":
                        N_in = N_temp["waned"] + N_temp["second_dose"]
                    else:
                        N_in = self.vaccine.get_num_eligible(
                            N,
                            A * L,
                            v_temp.v_name,
                            v_temp.v_in,
                            v_temp.v_out,
                            calendar[t],
                        )

                    ratio_S_N = np.array(
                        [
                            0 if N_in[i] == 0 else float(S_in[i] / N_in[i])
                            for i in range(len(N_in))
                        ]
                    ).reshape((A, L))

                    in_sum += (ratio_S_N * S_temp[v_temp.v_name]).astype(int)

            v_group.S = v_group.S + (np.array(in_sum - out_sum))
            S_after = np.zeros((5, 2))

        for v_group in self.vaccine_groups:
            S_after += v_group.S

        imbalance = np.abs(np.sum(S_before - S_after, axis=(0, 1)))

        # Note: add this to tests! But take this out of code
        # assert (imbalance < 1e-2).all(), (
        #     f"fPop inbalance in vaccine flow in between compartment S "
        #     f"{imbalance} at time {calendar[t]}, {t}"
        # )

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

    def get_binomial_transition_quantity(self, n, p):

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
