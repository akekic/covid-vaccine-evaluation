import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import concurrent.futures

from vaccination_policy import VaccinationPolicy, ObservedVaccinationPolicy
from causal_covid.run_scenario import multi_dimensional
from causal_covid.utils import day_to_week_matrix

logger = logging.getLogger(__name__)

AGE_GROUP_NAMES = [
    "0-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70-79",
    "80-89",
    "90+",
]


class SeverityFactorisation:
    subdir_name = "severity_factorisation"

    def __init__(
        self,
        factorisation_data_dir: Path,
        load_correction_factor: bool,
        vaccination_policy: Optional[VaccinationPolicy] = None,
        observed_vaccination_policy: Optional[ObservedVaccinationPolicy] = None,
        C_mat_param: Optional[int] = None,
        V1_eff: int = 70,
        V2_eff: int = 90,
        V3_eff: int = 95,
        draws: Optional[int] = None,
        influx: Optional[int] = None,
        waning_path: Optional[Path] = None,
        baseline_from_observations: bool = False,
    ) -> None:
        self.path = factorisation_data_dir
        if waning_path is not None:
            logger.info(f"Alternative waning path {waning_path} is used.")
        self.waning_path = waning_path
        self.load_correction_factor = load_correction_factor
        self.vaccination_policy = vaccination_policy
        self.C_mat_param = C_mat_param
        self.V1_eff = V1_eff
        self.V2_eff = V2_eff
        self.V3_eff = V3_eff
        self.draws = draws
        self.influx = influx
        self.baseline_from_observations = baseline_from_observations

        self.f_0, self.g = self._load_base_factors()

        (
            self.f_1,
            self.f_1_samples,
            self.infection_dynamics_df,
            self.infection_dynamics_samples,
            self.median_weekly_base_R_t,
            self.median_weekly_eff_R_t,
        ) = self._generate_infection_dynamics_simulation(
            factorisation_data_dir,
            generate_correction_factors=load_correction_factor,
            vaccination_policy=vaccination_policy,
            observed_vaccination_policy=observed_vaccination_policy,
            baseline_from_observations=baseline_from_observations,
        )  # [age, t]

    def _load_base_factors(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load the base factors f_0 and g from the file system.

        Returns
        -------
        f_0 : np.ndarray
            The overall time dependence f_0(T). Shape: [t]
        g : np.ndarray
            Risk factors g(V, A). Shape: [vaccination_status, age]
        """
        df_f = pd.read_csv(self.path / "time_dependence.csv")
        f_0 = df_f["f"].values  # [t]

        df_g = pd.read_csv(self.path / "risk_factors.csv")
        g = np.stack(
            [df_g["g0"].values, df_g["g1"].values, df_g["g2"].values, df_g["g3"].values]
        )  # [vaccination_status, age]

        df_vaccine_efficacy = pd.read_csv(
            self.path / "vaccine_efficacy_waning_data.csv"
            if self.waning_path is None
            else self.waning_path  # TODO: this is not great
        )
        self.vaccine_efficacy_params = df_vaccine_efficacy["vaccine_efficacy"].values

        return f_0, g

    def _generate_infection_dynamics_simulation(
        self,
        factorisation_data_dir: Path,
        generate_correction_factors: bool,
        vaccination_policy: Optional[VaccinationPolicy] = None,
        observed_vaccination_policy: Optional[ObservedVaccinationPolicy] = None,
        baseline_from_observations: bool = False,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        Union[pd.DataFrame, None],
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Generate the infection dynamics simulation.

        Parameters
        ----------
        factorisation_data_dir: Path
            Path to the factorisation data directory.
        generate_correction_factors: bool
            Whether to generate the correction factors and reporoduction numbers. If False, dummy values are returned.
        vaccination_policy: Optional[VaccinationPolicy]
        observed_vaccination_policy: Optional[ObservedVaccinationPolicy]
        baseline_from_observations: bool

        Returns
        -------
        correction_factors: ndarray
            The correction factors. Shape: [age, t]
        correction_factors_samples: ndarray
            The correction factors samples. Shape: [samples, age, t]
        df_infection_dynamics: pd.DataFrame
            Infection dynamics output.
        infection_dynamics_samples: ndarray
            Infection dynamics samples. Shape: [samples, age, t]
        median_weekly_base_R_t: ndarray or None
            Base reproduction number. Shape: [t, age]
        median_weekly_eff_R_t: ndarray or None
            Effective reproduction number. Shape: [t, age]
        """
        if generate_correction_factors:
            assert (
                vaccination_policy is not None
            ), "Need to pass vaccination policy to compute correction factor"
            if not baseline_from_observations:
                assert (
                    observed_vaccination_policy is not None
                ), "Need to pass observed vaccination policy to compute correction factor"
            assert (
                self.C_mat_param is not None
            ), "Need to pass C_mat_param to compute correction factor"
            assert (
                self.draws is not None
            ), "Need to pass draws to compute correction factor"
            assert (
                self.influx is not None
            ), "Need to pass influx to compute correction factor"

            # TODO: handle start/end week None
            logger.info("Computing infection dynamics for scenario policy ...")
            (
                df_scenario_infection,
                median_weekly_base_R_t,
                median_weekly_eff_R_t,
                predictive_trace_scenario,
            ) = self._compute_scenario_infection_data(
                vaccination_policy=vaccination_policy,
                C_mat_param=self.C_mat_param,
                V1_eff=self.V1_eff,
                V2_eff=self.V2_eff,
                V3_eff=self.V3_eff,
                draws=self.draws,
                influx=self.influx,
                waning_path=self.waning_path,
            )
            df_scenario_infection = self._add_total_infections(df_scenario_infection)
            infection_dynamics_samples = predictive_trace_scenario["weekly_cases"]

            if baseline_from_observations:
                logger.info("Using observations for correction factor baseline")
                df_observed_infection = (
                    SeverityFactorisation._load_observed_infection_data(
                        factorisation_data_dir
                    )
                )

                correction_factors_samples = None
            else:
                logger.info(
                    "Using infection dynamics output for correction factor baseline"
                )
                logger.info("Computing infection dynamics for observed policy ...")
                (
                    df_observed_infection,
                    _,
                    _,
                    predictive_trace_observed,
                ) = self._compute_scenario_infection_data(
                    vaccination_policy=observed_vaccination_policy,
                    C_mat_param=self.C_mat_param,
                    V1_eff=self.V1_eff,
                    V2_eff=self.V2_eff,
                    V3_eff=self.V3_eff,
                    draws=self.draws,
                    influx=self.influx,
                    waning_path=None,
                )
                df_observed_infection = self._add_total_infections(
                    df_observed_infection
                )

                correction_factors_samples = (
                    predictive_trace_scenario["weekly_cases"]
                    / predictive_trace_observed["weekly_cases"]
                ).swapaxes(-1, -2)
                logger.info("Infection dynamics scenario computation finished")

            df_infection_dynamics = df_observed_infection.merge(
                df_scenario_infection,
                on=["Sunday_date", "Age_group"],
                suffixes=("_observed", "_scenario"),
                how="outer",
            )

            df_infection_dynamics["f1"] = (
                df_infection_dynamics["total_infections_scenario"]
                / df_infection_dynamics["total_infections_observed"]
            )

            # evaluate infection dynamics result
            for a_name in AGE_GROUP_NAMES:
                df_tmp = df_infection_dynamics[
                    (df_infection_dynamics["Age_group"] == a_name)
                ]
                if df_tmp["f1"].isnull().values.any():
                    logger.warning(
                        f"Null values encountered in scenario infection numbers computed by"
                        f" infection dynamics model for age group {a_name}"
                    )
            df_infection_dynamics["f1"] = df_infection_dynamics["f1"].fillna(1)
            df_infection_dynamics = self._interpolate_missing_values(
                df_infection_dynamics
            )

            correction_factors = df_infection_dynamics.pivot_table(
                values="f1", columns="Age_group", index="Sunday_date"
            )[AGE_GROUP_NAMES].T.values
        else:
            logger.info(
                "Using default value 1 for infection dynamics correction factor"
            )
            df_infection_dynamics = None
            median_weekly_base_R_t = None
            median_weekly_eff_R_t = None
            correction_factors = np.ones((9, 53))  # TODO: get dimensions
            correction_factors_samples = np.ones((1000, 9, 53))  # TODO: get dimensions
            infection_dynamics_samples = np.zeros((1000, 9, 53))  # TODO: get dimensions
        return (
            correction_factors,
            correction_factors_samples,
            df_infection_dynamics,
            infection_dynamics_samples,
            median_weekly_base_R_t,
            median_weekly_eff_R_t,
        )

    @staticmethod
    def _add_total_infections(df: pd.DataFrame) -> pd.DataFrame:
        df_observed_infection_total = pd.DataFrame(
            df.groupby("Sunday_date")["total_infections"].sum()
        ).reset_index()
        df_observed_infection_total["Age_group"] = "total"
        df = pd.concat((df, df_observed_infection_total)).sort_values(
            ["Sunday_date", "Age_group"]
        )
        return df

    def _compute_scenario_infection_data(
        self,
        vaccination_policy: VaccinationPolicy,
        C_mat_param: int,
        V1_eff: int,
        V2_eff: int,
        V3_eff: int,
        draws: int,
        influx: int,
        waning_path: Optional[Path],
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        (
            median_cases,
            predictive_trace,
            model,
            infectiability_df,
        ) = self._run_infection_dynamics(
            U2=vaccination_policy.U_2_full,
            u3=vaccination_policy.u_3_full,
            C_mat_param=C_mat_param,
            V1_eff=V1_eff,
            V2_eff=V2_eff,
            V3_eff=V3_eff,
            draws=draws,
            influx=influx,
            waning_path=waning_path,
        )
        df_scenario_infection = (
            pd.DataFrame(
                np.mean(predictive_trace["weekly_cases"], axis=0),
                columns=median_cases.columns,
                index=median_cases.index,
            )
            .rename_axis("Sunday_date")
            .reset_index()
            .melt(
                id_vars=["Sunday_date"],
                var_name="Age_group",
                value_name="total_infections",
            )
        )
        df_scenario_infection_err_low = (
            pd.DataFrame(
                np.percentile(predictive_trace["weekly_cases"], 5, axis=0),
                columns=median_cases.columns,
                index=median_cases.index,
            )
            .rename_axis("Sunday_date")
            .reset_index()
            .melt(
                id_vars=["Sunday_date"],
                var_name="Age_group",
                value_name="total_infections_err_low",
            )
        )
        df_scenario_infection_err_high = (
            pd.DataFrame(
                np.percentile(predictive_trace["weekly_cases"], 95, axis=0),
                columns=median_cases.columns,
                index=median_cases.index,
            )
            .rename_axis("Sunday_date")
            .reset_index()
            .melt(
                id_vars=["Sunday_date"],
                var_name="Age_group",
                value_name="total_infections_err_high",
            )
        )
        df_scenario_infection = df_scenario_infection.merge(
            df_scenario_infection_err_low, on=["Sunday_date", "Age_group"]
        )
        df_scenario_infection = df_scenario_infection.merge(
            df_scenario_infection_err_high, on=["Sunday_date", "Age_group"]
        )
        transf_mat = day_to_week_matrix(
            model.sim_begin, model.sim_end, infectiability_df.index
        )
        median_weekly_base_R_t = (transf_mat.T / 7) @ np.median(
            predictive_trace["base_R_t"], axis=0
        )
        median_weekly_eff_R_t = (transf_mat.T / 7) @ np.median(
            predictive_trace["eff_R_t"], axis=0
        )
        return (
            df_scenario_infection,
            median_weekly_base_R_t,
            median_weekly_eff_R_t,
            predictive_trace,
        )

    @staticmethod
    def _run_infection_dynamics(
        U2,
        u3,
        C_mat_param: int,
        V1_eff: int,
        V2_eff: int,
        V3_eff: int,
        draws: int,
        influx: int,
        waning_path: Optional[Path] = None,
    ):
        # wrapping the execution in ProcessPoolExecutor to avoid memory issues
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                multi_dimensional,
                U2=U2,
                u3=u3,
                C_mat_param=C_mat_param,
                V1_eff=V1_eff,
                V2_eff=V2_eff,
                V3_eff=V3_eff,
                draws=draws,
                influx=influx,
                waning_file=waning_path,
            )
        median_cases, predictive_trace, model, infectiability_df = future.result()
        return median_cases, predictive_trace, model, infectiability_df

    @staticmethod
    def _load_observed_infection_data(path: Path) -> pd.DataFrame:
        df_observed_infection = pd.read_csv(
            path / "observed_infection_data.csv", parse_dates=["Sunday_date"]
        )
        df_observed_infection["total_infections"] = df_observed_infection[
            [
                "positive_unvaccinated",
                "positive_after_1st_dose",
                "positive_after_2nd_dose",
                "positive_after_3rd_dose",
            ]
        ].sum(axis=1)
        df_observed_infection = df_observed_infection.drop(
            columns=[
                "positive_unvaccinated",
                "positive_after_1st_dose",
                "positive_after_2nd_dose",
                "positive_after_3rd_dose",
            ]
        )
        return df_observed_infection

    def save(self, dir: Path) -> None:
        subdir = dir / self.subdir_name
        subdir.mkdir(parents=True, exist_ok=False)
        np.save(subdir / "f_0", self.f_0)
        np.save(subdir / "f_1", self.f_1)
        if self.infection_dynamics_df is not None:
            self.infection_dynamics_df.to_csv(
                subdir / "infection_dynamics.csv", index=False
            )
        if self.infection_dynamics_samples is not None:
            np.save(
                subdir / "infection_dynamics_samples", self.infection_dynamics_samples
            )
        if self.median_weekly_base_R_t is not None:
            np.save(subdir / "median_weekly_base_R_t", self.median_weekly_base_R_t)
        if self.median_weekly_base_R_t is not None:
            np.save(subdir / "median_weekly_eff_R_t", self.median_weekly_eff_R_t)
        np.save(subdir / "g", self.g)
        np.save(subdir / "vaccine_efficacy_params", self.vaccine_efficacy_params)

    def _ve(self, x):
        return self.vaccine_efficacy_params[x]

    @staticmethod
    def _interpolate_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        for age_group in AGE_GROUP_NAMES:
            df.loc[df["Age_group"] == age_group, "f1"] = (
                df.loc[df["Age_group"] == age_group, "f1"]
                .replace([-np.inf, np.inf], np.nan)
                .interpolate()
            )
        return df
