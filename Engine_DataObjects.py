###############################################################################

# Engine_DataObjects.py

###############################################################################

import json
from math import log

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

import copy

import datetime as dt

WEEKDAY = 1
WEEKEND = 2
HOLIDAY = 3
LONG_HOLIDAY = 4


class DataPrepConfig:
    date_format = "%m/%d/%y"
    base_path = Path(__file__).parent


class TimeSeriesManager:

    def __init__(self):
        self.dict_names = {}

    def create_fixed_time_series_from_monthdayyear_df(self,
                                                      date_column_name,
                                                      df,
                                                      simulation_datetimes):
        df[date_column_name] = pd.to_datetime(df[date_column_name], format=DataPrepConfig.date_format)
        for col in df.columns:
            if col != date_column_name:
                time_series = np.asarray(df[(df[date_column_name] >= simulation_datetimes[0]) &
                                            (df[date_column_name] < simulation_datetimes[-1])][col])
                self.dict_names[col] = time_series

    def create_fixed_time_series_from_monthdayyear_intervals(self,
                                                             name,
                                                             intervals,
                                                             simulation_datetimes):
        time_series = [
            any(
                dt.datetime.strptime(start, DataPrepConfig.date_format) <= d
                <= dt.datetime.strptime(end, DataPrepConfig.date_format)
                for start, end in intervals
            )
            for d in simulation_datetimes
        ]
        self.dict_names[name] = time_series


class Calendar:
    def __init__(self,
                 city_name,
                 calendar_filename,
                 start_date,
                 sim_length):
        date_and_day_type_df = pd.read_csv(DataPrepConfig.base_path / "instances" / f"{city_name}" / calendar_filename)
        date_and_day_type_df["Date"] = pd.to_datetime(date_and_day_type_df["Date"],
                                                      format=DataPrepConfig.date_format)

        self.date_and_day_type_df = date_and_day_type_df
        self.start = dt.datetime.strptime(start_date, DataPrepConfig.date_format)
        self.simulation_datetimes = [self.start + dt.timedelta(days=t) for t in range(sim_length)]
        self.dict_datetime_to_sim_time = {d: d_ix for (d_ix, d) in enumerate(self.simulation_datetimes)}
        self._day_type = np.asarray(
            self.date_and_day_type_df["DayType"][self.date_and_day_type_df["Date"] >= self.start])


class City:
    def __init__(
            self,
            city_name,
            calendar_instance,
            setup_filename,
            variant_filename,
            hospital_home_timeseries_filename,
            variant_prevalence_filename,
    ):
        self.city_name = city_name
        self.cal = calendar_instance
        self.path_to_data = DataPrepConfig.base_path / "instances" / f"{city_name}"

        self.load_setup_data(setup_filename)
        self.load_hosp_related_data(hospital_home_timeseries_filename)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load prevalence data
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Read the combined variant files instead of a separate file for each new variant:
        df_variant = pd.read_csv(str(self.path_to_data / variant_prevalence_filename))
        df_variant["date"] = pd.to_datetime(df_variant["date"])

        with open(self.path_to_data / variant_filename, "r") as input_file:
            variant_data = json.load(input_file)
        self.variant_pool = VariantPool(variant_data, df_variant)
        self.variant_start = df_variant["date"][0]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Define dimension variables & others
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Number of age and risk groups
        self.A = len(self.N)
        self.L = len(self.N[0])

        # Maximum simulation length
        self.T = 1 + (self.simulation_end_date - self.simulation_start_date).days

    def load_hosp_related_data(self, hospital_home_timeseries_filename):

        # hospital_home_timeseries should have 5+1 columns
        #   first column is "date" column
        # Subsequent columns correspond to data_varnames, in order

        data_varnames = ("real_IH_history",
                         "real_ICU_history",
                         "real_ToIHT_history",
                         "real_ToICUD_history",
                         "real_ToIYD_history")

        df_hosp = pd.read_csv(str(self.path_to_data / hospital_home_timeseries_filename))
        df_hosp["date"] = pd.to_datetime(df_hosp["date"])

        df_hosp = df_hosp[df_hosp["date"] <= self.simulation_end_date]
        df_hosp = df_hosp[df_hosp["date"] >= self.simulation_start_date]

        # If simulation starts before date that historical hospital data starts,
        #   add 0s to historical hospital data for those dates
        # Note this might mess up R^2 computations so be wary of interpreting the
        #   R^2 in this context
        if df_hosp["date"].iloc[0] > self.simulation_start_date:
            num_historical_data_missing_days = (df_hosp["date"].iloc[0] - self.simulation_start_date).days
        else:
            num_historical_data_missing_days = 0

        # in df_hosp, first column is "date" so start after that column
        for i in range(len(data_varnames)):
            timeseries = [0] * num_historical_data_missing_days + list(df_hosp[df_hosp.columns[i + 1]])
            setattr(self, data_varnames[i], timeseries)

    def load_setup_data(self, setup_filename):
        with open(str(self.path_to_data / setup_filename), "r") as input_file:
            data = json.load(input_file)
            # assert self.city_name == data["city_name"], "Data file does not match city."

            # LP note: smooth this later -- not all of these attributes
            #   are actually used and some of them are redundant
            for (k, v) in data.items():
                setattr(self, k, v)

            # Load demographics information
            self.N = np.array(data["population"])
            self.I0 = np.array(data["IY_ini"])

            # Load simulation dates
            self.simulation_start_date = pd.to_datetime(data["simulation_start_date"])
            self.simulation_end_date = pd.to_datetime(data["simulation_end_date"])


class TierInfo:
    def __init__(self, city_name, tier_filename):
        self.path_to_data = DataPrepConfig.base_path / "instances" / f"{city_name}"
        with open(str(self.path_to_data / tier_filename), "r") as tier_input:
            tier_data = json.load(tier_input)
            self.tier = tier_data["tiers"]


class Vaccine:
    """
    Vaccine class to define epidemiological characteristics, supply and fixed allocation schedule of vaccine.
    """

    def __init__(
            self,
            instance,
            city_name,
            vaccine_filename,
            booster_filename,
            vaccine_allocation_filename):

        self.path_to_data = DataPrepConfig.base_path / "instances" / f"{city_name}"

        with open(str(self.path_to_data / vaccine_filename), "r") as vaccine_input:
            vaccine_data = json.load(vaccine_input)

        vaccine_allocation_data = pd.read_csv(str(self.path_to_data / vaccine_allocation_filename))
        vaccine_allocation_data["vaccine_time"] = pd.to_datetime(vaccine_allocation_data["vaccine_time"])

        if booster_filename is not None:
            booster_allocation_data = pd.read_csv(str(self.path_to_data / booster_filename))
            booster_allocation_data["vaccine_time"] = pd.to_datetime(booster_allocation_data["vaccine_time"])
        else:
            booster_allocation_data = None

        self.effect_time = vaccine_data["effect_time"]
        self.second_dose_time = vaccine_data["second_dose_time"]
        self.beta_reduct = vaccine_data["beta_reduct"]
        self.tau_reduct = vaccine_data["tau_reduct"]
        self.pi_reduct = vaccine_data["pi_reduct"]
        self.instance = instance

        self.datetime_first_dose_effective = [
            time + pd.Timedelta(days=self.effect_time)
            for time in vaccine_allocation_data["vaccine_time"]]

        self.datetime_second_dose_effective = [
            time + pd.Timedelta(days=self.second_dose_time + self.effect_time)
            for time in self.datetime_first_dose_effective]

        self.vaccine_start_datetime = vaccine_allocation_data["vaccine_time"][0]

        self.vaccine_start_time = np.where(
            np.array(instance.cal.simulation_datetimes) == vaccine_allocation_data["vaccine_time"][0])[0]

        self.vaccine_allocation = self.define_supply(vaccine_allocation_data,
                                                     booster_allocation_data,
                                                     instance.N,
                                                     instance.A,
                                                     instance.L
                                                     )

        self.num_eligible_dict = {}

    def define_supply(self,
                      vaccine_allocation_data,
                      booster_allocation_data,
                      N,
                      A,
                      L):
        """
        Load vaccine supply and allocation data, and process them.
        Shift vaccine schedule for waiting vaccine to be effective,
            second dose and vaccine waning effect and also for booster dose.
        """

        # 10 of these age-risk groups (5 age groups, 2 risk groups)
        age_risk_columns = [
            column
            for column in vaccine_allocation_data.columns
            if "A" and "R" in column
        ]

        def create_allocation_item(datetime_instance,
                                   vac_assignment):
            return {
                "time": datetime_instance,
                "assignment": vac_assignment
            }

        def process_allocation(data,
                               datetime_list):
            vaccine_allocation_dict = {}
            for i in range(len(data)):
                vac_assignment = np.array(data[age_risk_columns].iloc[i]).reshape((A, L))
                vaccine_allocation_dict[datetime_list[i]] = vac_assignment
            return vaccine_allocation_dict

        # Process first dose allocation
        first_dose_allocation = process_allocation(vaccine_allocation_data, self.datetime_first_dose_effective)

        # Process second dose allocation
        second_dose_allocation = process_allocation(vaccine_allocation_data[:len(self.datetime_second_dose_effective)],
                                                    self.datetime_second_dose_effective)

        # Process booster allocation if data is available
        if booster_allocation_data is not None:
            booster_allocation = process_allocation(booster_allocation_data,
                                                    booster_allocation_data["vaccine_time"])

        return {
            "first_dose": first_dose_allocation,
            "second_dose": second_dose_allocation,
            "booster": booster_allocation
        }

    def create_num_eligible_dict(self,
                                 total_population,
                                 total_risk_gr,
                                 list_of_vaccine_groups,
                                 simulation_end_datetime):

        num_eligible_dict = {}

        for v_group in list_of_vaccine_groups:
            num_eligible_dict[v_group.v_name] = {}

        for v_group in list_of_vaccine_groups:
            event_datetime = self.vaccine_start_datetime
            N_in_cumulative = np.zeros((total_risk_gr, 1))
            N_out_cumulative = np.zeros((total_risk_gr, 1))
            while event_datetime <= simulation_end_datetime:
                for vaccine_type in v_group.v_in:
                    if event_datetime in self.vaccine_allocation[vaccine_type]:
                        N_in_cumulative += self.vaccine_allocation[vaccine_type][event_datetime].reshape((total_risk_gr, 1))
                for vaccine_type in v_group.v_out:
                    if event_datetime in self.vaccine_allocation[vaccine_type]:
                        N_out_cumulative += self.vaccine_allocation[vaccine_type][event_datetime].reshape((total_risk_gr, 1))
                event_datetime += pd.Timedelta(days=1)

                if v_group.v_name == "unvax":
                    num_eligible = total_population.reshape((total_risk_gr, 1)) - N_out_cumulative
                elif v_group.v_name == "waned":
                    # Waned compartment does not have incoming vaccine schedule but has outgoing scheduled vaccine. People
                    # enter waned compartment with binomial draw. This calculation would return negative value
                    num_eligible = np.zeros((total_risk_gr, 1))
                else:
                    num_eligible = N_in_cumulative - N_out_cumulative

                num_eligible_dict[v_group.v_name][event_datetime] = num_eligible

        self.num_eligible_dict = num_eligible_dict

        # Note: add this to tests! But take this out of code
        # assert (N_eligible > -1e-2).all(), (
        #     f"fPop negative eligible individuals for vaccination in vaccine group {vaccine_group_name}"
        #     f"{N_eligible} at time {date}"
        # )

class EpiParams:
    """
    A setup for the epidemiological parameters.
    Scenarios 6 corresponds to best guess parameters for UT group.
    """

    def __init__(self, params, rng):
        self.assign_random_epi_params(params, rng)
        self.assign_deterministic_epi_params(params)

    def assign_random_epi_params(self, epi_params_setup_info, rng):
        for k, v in epi_params_setup_info["random_params"].items():
            is_inverse = (k in epi_params_setup_info["list_names_inverse_params"])

            if rng is not None:
                setattr(self, k, self.get_random_sample_scalar_param(is_inverse, v["distribution_name"],
                                                                     v["distribution_params"], rng))
            else:
                setattr(self, k, v["deterministic_val"])

    def assign_deterministic_epi_params(self, epi_params_setup_info):
        for k, v in epi_params_setup_info["deterministic_params"].items():
            is_inverse = k in epi_params_setup_info["list_names_inverse_params"]
            value = np.array(v) if isinstance(v, list) else v
            setattr(self, k, 1 / value if is_inverse else value)

    def save_additional_initial_values_epi_params(self):

        # Important for updating epi params under variant

        self.alpha_gamma_ICU0 = self.alpha_gamma_ICU
        self.alpha_IH0 = self.alpha_IH
        self.alpha_mu_ICU0 = self.alpha_mu_ICU
        self.alpha_IYD0 = self.alpha_IYD
        self.sigma_E0 = copy.deepcopy(self.sigma_E)

    def get_random_sample_scalar_param(self,
                                       is_inverse,
                                       distribution_name,
                                       distribution_params,
                                       rng):
        random_distribution_function_name = getattr(rng, distribution_name)
        result = random_distribution_function_name(*distribution_params)

        return np.squeeze(1 / result) if is_inverse else np.squeeze(result)

    def setup_base_params(self):

        # See Yang et al. (2021) and Arslan et al. (2021)

        self.beta = self.beta0  # Unmitigated transmission rate
        self.YFR = self.IFR / self.tau  # symptomatic fatality ratio (%)
        self.pIH0 = self.pIH  # percent of patients going directly to general ward
        self.YHR0 = copy.deepcopy(self.YHR)  # % of symptomatic infections that go to hospital
        self.YHR_overall0 = copy.deepcopy(self.YHR_overall)

        # if gamma_IH and mu are lists, reshape them for right dimension
        self.gamma_IH0 = self.gamma_IH0.reshape(self.gamma_IH0.size, 1)
        self.etaICU = self.etaICU.reshape(self.etaICU.size, 1)
        self.etaICU0 = self.etaICU.copy()
        self.gamma_ICU0 = self.gamma_ICU0.reshape(self.gamma_ICU0.size, 1)
        self.mu_ICU0 = self.mu_ICU0.reshape(self.mu_ICU0.size, 1)
        self.HICUR0 = self.HICUR

        self.save_additional_initial_values_epi_params()

    @property
    def gamma_ICU(self):
        """
        Adjust hospital dynamics according to real data with self.alpha_ params.
        See Haoxiang et al. or Arslan et al. for more detail.
        This one increase the rate of recovering from ICU.
        """
        return self.gamma_ICU0 * (1 + self.alpha_gamma_ICU)

    @property
    def gamma_IH(self):
        """ This one decrease the rate of recovering from IH."""
        return self.gamma_IH0 * (1 - self.alpha_IH)

    @property
    def mu_ICU(self):
        """ This one increase the rate of death from ICU. """
        return self.mu_ICU0 * (1 + self.alpha_mu_ICU)

    def update_icu_all(self, t, time_series_manager):

        try:
            self.pIH = time_series_manager.dict_names["pIH"][t]
        except:
            self.pIH = self.pIH0

        try:
            self.HICUR = time_series_manager.dict_names["HICUR"][t]
        except:
            self.HICUR = self.HICUR0

        try:
            self.etaICU = self.etaICU0.copy() / time_series_manager.dict_names["etaICU"][t]
        except:
            self.etaICU = self.etaICU0.copy()

    @property
    def omega_P(self):
        """ infectiousness of pre-symptomatic relative to symptomatic """
        return np.array(
            [
                (
                        self.tau
                        * self.omega_IY
                        * (
                                self.YHR_overall[a] / self.Eta[a]
                                + (1 - self.YHR_overall[a]) / self.gamma_IY
                        )
                        + (1 - self.tau) * self.omega_IA / self.gamma_IA
                )
                / (self.tau * self.omega_IY / self.rho_Y + (1 - self.tau) * self.omega_IA / self.rho_A)
                * self.pp
                / (1 - self.pp)
                for a in range(len(self.YHR_overall))
            ]
        )

    @property
    def omega_PA(self):
        """ infectiousness of pre-asymptomatic individuals relative to IA for age-risk group a, r """
        return self.omega_IA * self.omega_P

    @property
    def omega_PY(self):
        """ infectiousness of pre-symptomatic individuals relative to IY for age-risk group a, r"""
        return self.omega_IY * self.omega_P

    @property
    def pi(self):
        """ rate-adjusted proportion of symptomatic individuals who go to the hospital for age-risk group a, r """
        return np.array(
            [
                self.YHR[a]
                * self.gamma_IY
                / (self.Eta[a] + (self.gamma_IY - self.Eta[a]) * self.YHR[a])
                for a in range(len(self.YHR))
            ]
        )

    @property
    def nu(self):
        """ rate-adjusted proportion of general ward patients transferred to ICU for age group a """
        return self.gamma_IH * self.HICUR / (self.etaICU + (self.gamma_IH - self.etaICU) * self.HICUR)

    @property
    def HFR(self):
        """ symptomatic fatality ratio divided by symptomatic hospitalization rate """
        return self.YFR / self.YHR

    @property
    def nu_ICU(self):
        return self.gamma_ICU0 * self.ICUFR / (self.mu_ICU0 + (self.gamma_ICU0 - self.mu_ICU0) * self.ICUFR)

    def effective_phi(self, school, cocooning, social_distance, demographics, day_type):
        """
        school (int): yes (1) / no (0) schools are closed
        cocooning (float): percentage of transmission reduction [0,1]
        social_distance (int): percentage of social distance (0,1)
        demographics (ndarray): demographics by age and risk group
        day_type (int): 1 Weekday, 2 Weekend, 3 Holiday, 4 Long Holiday
        """

        A = len(demographics)  # number of age groups
        L = len(demographics[0])  # number of risk groups
        d = demographics  # A x L demographic data
        phi_all_extended = np.zeros((A, L, A, L))
        phi_school_extended = np.zeros((A, L, A, L))
        phi_work_extended = np.zeros((A, L, A, L))
        for a, b in product(range(A), range(A)):
            phi_ab_split = np.array(
                [
                    [d[b, 0], d[b, 1]],
                    [d[b, 0], d[b, 1]],
                ]
            )
            phi_ab_split = phi_ab_split / phi_ab_split.sum(1)
            phi_ab_split = 1 + 0 * phi_ab_split / phi_ab_split.sum(1)
            phi_all_extended[a, :, b, :] = self.phi_all[a, b] * phi_ab_split
            phi_school_extended[a, :, b, :] = self.phi_school[a, b] * phi_ab_split
            phi_work_extended[a, :, b, :] = self.phi_work[a, b] * phi_ab_split

        # Apply school closure and social distance
        # Assumes 95% reduction on last age group and high risk cocooning

        if day_type == 1:  # Weekday
            phi_age_risk = (1 - social_distance) * (
                    phi_all_extended - school * phi_school_extended
            )
            if cocooning > 0:
                phi_age_risk_copy = phi_all_extended - school * phi_school_extended
        elif day_type == 2 or day_type == 3:  # is a weekend or holiday
            phi_age_risk = (1 - social_distance) * (
                    phi_all_extended - phi_school_extended - phi_work_extended
            )
            if cocooning > 0:
                phi_age_risk_copy = (
                        phi_all_extended - phi_school_extended - phi_work_extended
                )
        else:
            phi_age_risk = (1 - social_distance) * (
                    phi_all_extended - phi_school_extended
            )
            if cocooning > 0:
                phi_age_risk_copy = phi_all_extended - phi_school_extended
        if cocooning > 0:
            # High risk cocooning and last age group cocooning
            phi_age_risk[:, 1, :, :] = (1 - cocooning) * phi_age_risk_copy[:, 1, :, :]
            phi_age_risk[-1, :, :, :] = (1 - cocooning) * phi_age_risk_copy[-1, :, :, :]
        # assert (phi_age_risk >= 0).all()
        return phi_age_risk


class VariantPool:
    """
    A class that contains all the variant of concerns.
    """

    def __init__(self, variants_data: list, variants_prev: list):
        """
        :param variants_data: list of updates for each variant.
        :param variants_prev: prevalence of each variant of concern.
        """
        self.variants_data = variants_data
        self.variants_prev = variants_prev
        self.variant_names = variants_data["variant_names"]
        for (k, v) in self.variants_data["epi_params_under_variants"]["immune_evasion"].items():
            # calculate the rate of exponential immune evasion according to half-life (median) value:
            v["immune_evasion_max"] = log(2) / (v["half_life"] * 30) if v["half_life"] != 0 else 0
            v["start_date"] = pd.to_datetime(v["start_date"])
            v["peak_date"] = pd.to_datetime(v["peak_date"])
            v["end_date"] = pd.to_datetime(v["end_date"])

    def get_total_variant_prevalence(self, t_since_variant):
        return self.variants_prev.iloc[t_since_variant]["delta"] + self.variants_prev.iloc[t_since_variant]["omicron"]

    def get_sigma_E_under_variant(self, sigma_E, t_since_variant):
        total_variant_prevalence = self.get_total_variant_prevalence(t_since_variant)

        return sum(1 / (1/sigma_E - self.variants_data["epi_params_under_variants"]["sigma_E"][variant_name]) * \
               self.variants_prev.iloc[t_since_variant][variant_name] for variant_name in self.variant_names) + \
               (1 - total_variant_prevalence) * sigma_E

    def get_epi_params_under_variants(self, t_since_variant: int):
        """
        update epi parameters and vaccine parameters according to prevalence of different variances.
        :param t: current date
        :param sigma_E: current sampled sigma_E value in the simulation.
        :return: new set of params and total variant prev.
        """

        epi_params_under_variants = {}


        for (epi_param_name, epi_dict) in self.variants_data["epi_params_under_variants"].items():
            total_variant_prevalence = self.get_total_variant_prevalence(t_since_variant)
            if epi_param_name == "sigma_E":
                # sigma_E is computed differently than the other epi_params_under_variants variables
                pass
            elif epi_param_name == "immune_evasion":
                pass
            else:
                # For other parameters calculate the change in the value as a coefficent:
                epi_params_under_variants[epi_param_name] = 1 + sum(self.variants_prev.iloc[t_since_variant][variant_name] *
                                                              (self.variants_data["epi_params_under_variants"][epi_param_name][variant_name] - 1)
                                                              for variant_name in self.variant_names)

        return epi_params_under_variants

    def get_vaccine_params_under_variants(self, t_since_variant: int):

        vaccine_params_under_variants = {}

        # Calculate the new vaccine efficacy according to the variant values:
        for (vaccine_param_name, vaccine_dict) in self.variants_data["vaccine_params_under_variants"].items():
            for (vaccine_group_name, vaccine_group_dict) in vaccine_dict.items():
                vaccine_params_under_variants[(vaccine_param_name, vaccine_group_name)] = \
                    sum(self.variants_prev.iloc[t_since_variant][variant_name] *
                        vaccine_group_dict[variant_name] for variant_name in self.variant_names)

        return vaccine_params_under_variants

    def immune_evasion(self, immune_evasion_base: float, t):
        """
        I was planning to read the immune evasion value from the variant csv file, but we decide to run lsq on the
        immune evasion function, so I am integrating the piecewise linear function into the code.

        We assume the immunity evade with exponential rate.
        Calculate the changing immune evasion rate according to variant prevalence.
        Assume that the immune evasion follows a piecewise linear shape.
        It increases as the prevalence of variant increases and peak
        and starts to decrease.

        I assume the immune evasion functions of different variants do not overlap.

        half_life: half-life of the vaccine or natural infection induced protection.
        half_life_base: half-life of base level of immune evasion before the variant
        start_date: the date the immune evasion starts to increase
        peak_date: the date the immune evasion reaches the maximum level.

        :param immune_evasion_base: base immune evasion rate before the variant.
        :param t: current iterate
        :return: the immune evasion rate for a particular date.
        """
        for (k, v) in self.variants_data["epi_params_under_variants"]["immune_evasion"].items():
            if v["start_date"] <= t <= v["peak_date"]:
                days = (v["peak_date"] - v["start_date"]).days
                return (t - v["start_date"]).days * (
                        v["immune_evasion_max"] - immune_evasion_base) / days + immune_evasion_base
            elif v["peak_date"] <= t <= v["end_date"]:
                days = (v["end_date"] - v["peak_date"]).days
                return (v["end_date"] - t).days * (
                        v["immune_evasion_max"] - immune_evasion_base) / days + immune_evasion_base

        return immune_evasion_base

