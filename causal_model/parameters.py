import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd

from pathlib import Path
from itertools import product

logger = logging.getLogger(__name__)


class Parameters:
    """Class to store parameters."""
    subdir_name = "parameters"

    def __init__(
        self,
        path: Path,
        start_week: Optional[datetime.date] = None,
        end_week: Optional[datetime.date] = None,
    ) -> None:
        self.start_week = (
            np.datetime64(start_week)
            if start_week is not None
            else np.datetime64(datetime.date(year=2020, month=12, day=20))
        )
        assert self.start_week == np.datetime64(
            datetime.date(year=2020, month=12, day=20)
        ), "Start weeek is not set to start of vaccination period"
        self.end_week = (
            np.datetime64(end_week)
            if end_week is not None
            else np.datetime64(datetime.date(year=2021, month=12, day=19))
        )

        df_pop = pd.read_csv(path / "population_data.csv")
        self.P_a = df_pop["Population_share"].values
        self.D_a = df_pop["Population_size"].values

        self.week_dates = pd.read_csv(
            path / "time_dependence.csv", parse_dates=["Sunday_date"]
        )["Sunday_date"].values
        self.weeks = np.arange(len(self.week_dates))
        self.weeks_extended = np.arange(len(self.weeks) + 1)

        self.week_dates_scenario = self.week_dates[
            (self.start_week <= self.week_dates) & (self.week_dates <= self.end_week)
        ]
        self.weeks_scenario = np.arange(len(self.week_dates_scenario))
        self.weeks_scenario_extended = np.arange(len(self.weeks_scenario) + 1)

        self.P_t = 1 / len(self.weeks_scenario)

        self.vaccination_statuses = np.arange(4)
        self.age_groups = np.arange(9)
        self.age_group_names = df_pop["Age_group"].unique().astype(str)

        self.observed_vaccinations = self._load_vaccination_data(path)  # [age, t, dose]
        self.observed_severity = self._load_observed_severity_data(
            path
        )  # [age, t, vaccination_status]

        self.vaccine_acceptance_rate = self._load_vaccine_acceptance_rate(
            path
        )  # [age, dose]

        # TODO: pass gaps via input data
        self.l = 3  # noqa  # min. gap between 1st and 2nd vaccination
        self.k = 12  # min. gap between 2nd and 3rd vaccination

    @staticmethod
    def _load_vaccination_data(path: Path) -> np.ndarray:
        # TODO: check if weeks and ages are sorted in df
        df_vac = pd.read_csv(path / "vaccination_data.csv")
        age_groups = df_vac["Age_group"].unique()
        weeks = df_vac["Sunday_date"].unique()
        observed_vaccinations = np.zeros((len(age_groups), len(weeks), 3))
        for a, t in product(np.arange(len(age_groups)), np.arange(len(weeks))):
            observed_vaccinations[a, t, 0] = df_vac.loc[
                (df_vac["Sunday_date"] == weeks[t])
                & (df_vac["Age_group"] == age_groups[a]),
                "1st_dose",
            ]
            observed_vaccinations[a, t, 1] = df_vac.loc[
                (df_vac["Sunday_date"] == weeks[t])
                & (df_vac["Age_group"] == age_groups[a]),
                "2nd_dose",
            ]
            observed_vaccinations[a, t, 2] = df_vac.loc[
                (df_vac["Sunday_date"] == weeks[t])
                & (df_vac["Age_group"] == age_groups[a]),
                "3rd_dose",
            ]
        return observed_vaccinations

    @staticmethod
    def _load_observed_severity_data(path: Path) -> np.ndarray:
        df_sev = pd.read_csv(path / "observed_severity_data.csv")
        age_groups = df_sev["Age_group"].unique()
        weeks = df_sev["Sunday_date"].unique()
        observed_severity = np.zeros((len(age_groups), len(weeks), 4))
        for a, t in product(np.arange(len(age_groups)), np.arange(len(weeks))):
            observed_severity[a, t, 0] = df_sev.loc[
                (df_sev["Sunday_date"] == weeks[t])
                & (df_sev["Age_group"] == age_groups[a]),
                "hosp_unvaccinated_rel",
            ]
            observed_severity[a, t, 1] = df_sev.loc[
                (df_sev["Sunday_date"] == weeks[t])
                & (df_sev["Age_group"] == age_groups[a]),
                "hosp_after_1st_dose_rel",
            ]
            observed_severity[a, t, 2] = df_sev.loc[
                (df_sev["Sunday_date"] == weeks[t])
                & (df_sev["Age_group"] == age_groups[a]),
                "hosp_after_2nd_dose_rel",
            ]
            observed_severity[a, t, 3] = df_sev.loc[
                (df_sev["Sunday_date"] == weeks[t])
                & (df_sev["Age_group"] == age_groups[a]),
                "hosp_after_3rd_dose_rel",
            ]
        return np.nan_to_num(observed_severity)

    def _load_vaccine_acceptance_rate(self, path: Path) -> np.ndarray:
        df_acc = pd.read_csv(path / "vaccine_acceptance_data.csv")
        return df_acc[
            [
                "1st_dose_acceptance_rate",
                "2nd_dose_acceptance_rate",
                "3rd_dose_acceptance_rate",
            ]
        ].to_numpy()

    def save(self, dir: Path) -> None:
        subdir = dir / self.subdir_name
        subdir.mkdir(parents=True, exist_ok=False)
        np.save(subdir / "observed_vaccinations", self.observed_vaccinations)
        np.save(subdir / "P_a", self.P_a)
        np.save(subdir / "D_a", self.D_a)
        np.save(subdir / "P_t", self.P_t)
        np.save(subdir / "weeks", self.weeks)
        np.save(subdir / "weeks_extended", self.weeks_extended)
        np.save(subdir / "week_dates", self.week_dates)
        np.save(subdir / "weeks_scenario", self.weeks_scenario)
        np.save(subdir / "weeks_scenario_extended", self.weeks_scenario_extended)
        np.save(subdir / "week_dates_scenario", self.week_dates_scenario)
        np.save(subdir / "age_groups", self.age_groups)
        np.save(subdir / "age_group_names", self.age_group_names)
        np.save(subdir / "vaccination_statuses", self.vaccination_statuses)
