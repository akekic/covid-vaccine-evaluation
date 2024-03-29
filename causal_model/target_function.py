import logging
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import multiprocessing as mp

from itertools import product
from tqdm import tqdm

from causal_model.severity_factorisation import SeverityFactorisation
from causal_model.parameters import Parameters
from vaccination_policy.vaccination_policy import VaccinationPolicy

logger = logging.getLogger(__name__)


class TargetFunction:
    """
    This class contains the core of the target function computation.

    The target function is the expected number of severe cases in the population for a given
    vaccine allocation policy.
    """
    subdir_name = "target_function"

    def __init__(
        self,
        params: Parameters,
        severity_factorisation: SeverityFactorisation,
        waning_states: Optional[list[int]] = None,
    ) -> None:
        """
        Parameters
        ----------
        params : Parameters
            General parameters of the simulation.
        severity_factorisation : SeverityFactorisation
            Factors making up the severity mechanism. Also contains the infection dynamics simulation.
        waning_states : list[int], optional
            List of vaccination states for which waning is considered, by default we only consider waning after
            the second vaccination.
        """
        self.waning_states = (
            [
                2,
            ]
            if waning_states is None
            else waning_states
        )
        assert all(isinstance(x, int) for x in self.waning_states)
        self.age_groups = params.age_groups
        self.weeks_scenario = params.weeks_scenario
        self.vaccination_statuses = params.vaccination_statuses
        self.weeks_scenario_extended = params.weeks_scenario_extended
        self.severity_factorisation = severity_factorisation
        self.P_t = params.P_t
        self.P_a = params.P_a
        self.D_a = params.D_a
        self.observed_severity = params.observed_severity

    def waning_function(self, t, vac_time, vac_state):
        """
        Waning function for the vaccine efficacy.

        Parameters
        ----------
        t : int
            Time step (week index).
        vac_time : int
            Time step (week index) of the vaccination.
        vac_state : int
            Vaccination state of the individual (0, 1, 2 or 3).
        """
        if vac_state in self.waning_states:
            x = t - vac_time
            if vac_state == 1:
                expected_ve_0 = float(self.severity_factorisation.V1_eff) / 100
            elif vac_state == 2:
                expected_ve_0 = float(self.severity_factorisation.V2_eff) / 100
            elif vac_state == 3:
                expected_ve_0 = float(self.severity_factorisation.V3_eff) / 100
            else:
                raise ValueError(
                    f"Invalid value {vac_state} for v - expected 1, 2 or 3"
                )

            s = expected_ve_0 / self.severity_factorisation._ve(0)
            if isinstance(x, np.ndarray):
                if x.size == 0:
                    return np.empty_like(x)
                else:
                    return (
                        1 - s * np.vectorize(self.severity_factorisation._ve)(x)
                    ) / (1.0 - s * self.severity_factorisation._ve(0))
            else:
                return (1.0 - s * self.severity_factorisation._ve(x)) / (
                    1.0 - s * self.severity_factorisation._ve(0)
                )
        else:
            return 1

    def __call__(
        self,
        vaccination_policy: VaccinationPolicy,
        ignore_infection_dynamics: bool = False,
        n_samples: int = 0,
        n_workers: int = 1,
        *args,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the target function for a given vaccination policy.

        Parameters
        ----------
        vaccination_policy : VaccinationPolicy
            Vaccination policy to evaluate.
        ignore_infection_dynamics : bool, optional
            If True, the infection dynamics are not simulated and we assume the correction factor is 1
            for all age groups and time steps, by default False.
        n_samples : int, optional
            Number of samples to take from the infection dynamics simulation, by default 0
            (only compute the mean estimate).
        n_workers : int, optional
            Number of workers to use. The target function is computed for each sample of the infection dynamics.
            This computation is parallelised over the samples, by default 1.

        Returns
        -------
        s_t_a_v : np.ndarray
            Number of severe cases for each age group, time step and vaccination state (mean estimate).
        s_t_a_v_samples : np.ndarray
            Number of severe cases for each age group, time step and vaccination state
            (for each sample of the infection dynamics simulation).

        """
        if ignore_infection_dynamics:
            logger.info("Computing target function with ignored infection dynamics ...")
        else:
            logger.info("Computing target function ...")
        f_1 = (
            np.ones_like(self.severity_factorisation.f_1)
            if ignore_infection_dynamics
            else self.severity_factorisation.f_1
        )

        logger.info("Compute mean estimate for target function ...")
        s_t_a_v = self._core_call(
            weeks_scenario=self.weeks_scenario,
            age_groups=self.age_groups,
            vaccination_statuses=self.vaccination_statuses,
            weeks_scenario_extended=self.weeks_scenario_extended,
            U_2=vaccination_policy.U_2,
            u_3=vaccination_policy.u_3,
            g=self.severity_factorisation.g,
            P_t=self.P_t,
            f_0=self.severity_factorisation.f_0,
            f_1=f_1,
            population=self.D_a.sum(),
            waning_function=self.waning_function,
        )

        if n_samples > 0:
            logger.info(
                f"Compute {n_samples} target function samples using {n_workers} workers ..."
            )
            with mp.Pool(n_workers) as pool:
                jobs = [
                    pool.apply_async(
                        self._core_call,
                        kwds={
                            "weeks_scenario": self.weeks_scenario,
                            "age_groups": self.age_groups,
                            "vaccination_statuses": self.vaccination_statuses,
                            "weeks_scenario_extended": self.weeks_scenario_extended,
                            "U_2": vaccination_policy.U_2,
                            "u_3": vaccination_policy.u_3,
                            "g": self.severity_factorisation.g,
                            "P_t": self.P_t,
                            "f_0": self.severity_factorisation.f_0,
                            "f_1": f_1_sample,
                            "population": self.D_a.sum(),
                            "waning_function": self.waning_function,
                        },
                    )
                    for f_1_sample in self.severity_factorisation.f_1_samples[
                        :n_samples
                    ]
                ]
                results = []
                for job in tqdm(jobs):
                    results.append(job.get())
                s_t_a_v_samples = np.stack(results)
        else:
            s_t_a_v_samples = None

        return s_t_a_v, s_t_a_v_samples

    @staticmethod
    def _core_call(
        weeks_scenario: np.ndarray,
        age_groups: np.ndarray,
        vaccination_statuses: np.ndarray,
        weeks_scenario_extended: np.ndarray,
        U_2: np.ndarray,
        u_3: np.ndarray,
        g: np.ndarray,
        P_t: float,
        f_0: np.ndarray,
        f_1: np.ndarray,
        population: float,
        waning_function: Callable,
    ) -> np.ndarray:
        """
        Compute the target function for a given vaccination policy and infection dynamics sample.

        Parameters
        ----------
        weeks_scenario : np.ndarray
            Time steps of the scenario.
        age_groups : np.ndarray
            Age groups of the scenario.
        vaccination_statuses : np.ndarray
            One week longer than weeks_scenario.
        weeks_scenario_extended : np.ndarray
            Time steps of the extended scenario.
        U_2 : np.ndarray
            Parametrisation of the vaccination policy (first and second doses).
        u_3 : np.ndarray
            Parametrisation of the vaccination policy (third dose).
        g : np.ndarray
            Risk factors g(V, A) of the severity factorisation.
        P_t : float
            Probability of a given time, typically 1/ number of time steps.
        f_0 : np.ndarray
            Overall time dependence f_0(T) of the severity factorisation.
        f_1 : np.ndarray
            Infection dynamics correction factor f_1(A, T) of the severity factorisation.
        population : float
            Population size.
        waning_function : Callable
            Waning function.

        Returns
        -------
        s_t_a_v : np.ndarray
            Number of severe cases for each age group, time step and vaccination state.
        """
        s_t_a_v = np.zeros(
            (
                len(weeks_scenario),
                len(age_groups),
                len(vaccination_statuses),
            )
        )

        for t, a in product(weeks_scenario, age_groups):
            # unvaccinated
            tmp_0 = 0
            for t1, t2, t3 in product(
                weeks_scenario_extended[t + 1 :],
                weeks_scenario_extended[t + 1 :],
                weeks_scenario_extended[:],
            ):
                tmp_0 += U_2[a, t1, t2] * u_3[a, t2, t3]
            tmp_0 *= g[0, a] * P_t * f_0[t] * f_1[a, t] / population
            s_t_a_v[t, a, 0] = tmp_0

            # after 1st dose
            tmp_1 = 0
            for t1, t2, t3 in product(
                weeks_scenario_extended[: t + 1],
                weeks_scenario_extended[t + 1 :],
                weeks_scenario_extended[:],
            ):
                waning_factor_1 = waning_function(t, t1, vac_state=1)
                tmp_1 += U_2[a, t1, t2] * u_3[a, t2, t3] * waning_factor_1
            tmp_1 *= g[1, a] * P_t * f_0[t] * f_1[a, t] / population
            s_t_a_v[t, a, 1] = tmp_1

            # after 2nd dose
            tmp_2 = 0
            for t1, t2, t3 in product(
                weeks_scenario_extended[: t + 1],
                weeks_scenario_extended[: t + 1],
                weeks_scenario_extended[t + 1 :],
            ):
                waning_factor_2 = waning_function(t, t2, vac_state=2)
                tmp_2 += U_2[a, t1, t2] * u_3[a, t2, t3] * waning_factor_2
            tmp_2 *= g[2, a] * P_t * f_0[t] * f_1[a, t] / population
            s_t_a_v[t, a, 2] = tmp_2

            # after 3rd dose
            tmp_3 = 0
            for t1, t2, t3 in product(
                weeks_scenario_extended[: t + 1],
                weeks_scenario_extended[: t + 1],
                weeks_scenario_extended[: t + 1],
            ):
                waning_factor_3 = waning_function(t, t3, vac_state=3)
                tmp_3 += U_2[a, t1, t2] * u_3[a, t2, t3] * waning_factor_3
            tmp_3 *= g[3, a] * P_t * f_0[t] * f_1[a, t] / population
            s_t_a_v[t, a, 3] = tmp_3

        return s_t_a_v

    def save(self, dir: Path) -> None:
        subdir = dir / self.subdir_name
        subdir.mkdir(parents=True, exist_ok=False)
