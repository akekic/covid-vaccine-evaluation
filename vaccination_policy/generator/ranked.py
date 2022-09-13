from __future__ import annotations

import numbers
from typing import Union, TYPE_CHECKING, Optional

import numpy as np
from numpy import typing as npt

from vaccination_policy import VaccinationPolicy, FixedDosesVaccinationPolicy
from .available_vaccinations import UnassignedVaccinations
from .base import VaccinationPolicyGenerator

if TYPE_CHECKING:
    from causal_model import Parameters, SeverityFactorisation


class RankedVaccinationPolicyGenerator(VaccinationPolicyGenerator):
    def __init__(
        self,
        params: Parameters,
        ranking: Union[str, list[tuple[int, int]]] = "elderly_first",
        vaccine_acceptance_rate: Union[
            float, str, npt.ArrayLike
        ] = 0.8,  # options: 'observed', array, number
        constraints_per_dose: bool = False,
        severity_factorisation: Optional[SeverityFactorisation] = None,
    ):
        super().__init__(params)

        self.severity_factorisation = severity_factorisation
        self.constraints_per_dose = constraints_per_dose
        self.gap_ranking = np.arange(self.params.l, 40)

        self.ranking = self._make_ranking(ranking)

        self.vaccine_acceptance_rate = self._make_vaccine_acceptance_rate(
            vaccine_acceptance_rate
        )

    def generate(self) -> VaccinationPolicy:
        U_2, u_3 = np.zeros(self.parametrisation_dims), np.zeros(
            self.parametrisation_dims
        )

        unassigned_vaccinations = UnassignedVaccinations(
            initially_available_vaccinations=np.copy(self.params.observed_vaccinations),
            constraints_per_dose=self.constraints_per_dose,
        )

        for t in self.params.weeks:
            for a, v in self.ranking:
                U_2, u_3, unassigned_vaccinations = self._assign_vaccinations(
                    U_2, u_3, unassigned_vaccinations, t, a, v
                )

        for a, _ in self.ranking[self.ranking[:, 1] == 2]:
            for t2 in self.params.weeks:
                # only twice vaccinated
                u_3[a, t2, -1] = 1 - u_3[a, t2, :].sum()

        # these are dummy values which make the summation in the target function simpler
        # it does not mean that there are people who got their 3rd but not their 2nd shot
        u_3[:, -1, :] = 1.0 / (len(self.params.weeks) + 1)

        for a, _ in self.ranking[self.ranking[:, 1] == 2]:
            for t1 in self.params.weeks:
                # only once vaccinated
                U_2, unassigned_vaccinations = self._assign_first_only_doses(
                    U_2, unassigned_vaccinations, a, t1
                )

            # completely unvaccinated
            U_2[a, -1, -1] = self.params.D_a[a] - U_2[a, ...].sum()

        # assert unassigned_vaccinations.sum() == 0
        policy = FixedDosesVaccinationPolicy(self.params, U_2=U_2, u_3=u_3)
        policy.validate()
        return policy

    def _make_vaccine_acceptance_rate(
        self, vaccine_acceptance_rate: Union[float, str, npt.ArrayLike]
    ) -> npt.ArrayLike:
        if isinstance(vaccine_acceptance_rate, numbers.Number):
            return vaccine_acceptance_rate * np.ones(
                (len(self.params.age_groups), len(self.params.vaccination_statuses))
            )
        elif isinstance(vaccine_acceptance_rate, str):
            if vaccine_acceptance_rate == "observed":
                return self.params.vaccine_acceptance_rate.copy()
            elif vaccine_acceptance_rate == "observed_relaxed":
                var = self.params.vaccine_acceptance_rate.copy()
                var[:, 2] += 0.025
                return var
            else:
                raise KeyError(
                    f"unsupported vaccine acceptance mode {vaccine_acceptance_rate}"
                )
        elif isinstance(vaccine_acceptance_rate, npt.ArrayLike):
            assert vaccine_acceptance_rate.shape == (
                self.params.age_groups,
                self.params.vaccination_statuses,
            )
            return vaccine_acceptance_rate
        else:
            raise KeyError(
                f"unsupported vaccine acceptance mode type {type(vaccine_acceptance_rate)}"
            )

    def _make_ranking(
        self, ranking: Union[str, list[tuple[int, int]]]
    ) -> npt.ArrayLike:
        if isinstance(ranking, str):
            if ranking == "young_first":
                return np.array(
                    [
                        (0, 2),
                        (1, 2),
                        (2, 2),
                        (3, 2),
                        (4, 2),
                        (5, 2),
                        (6, 2),
                        (7, 2),
                        (8, 2),
                        (0, 3),
                        (1, 3),
                        (2, 3),
                        (3, 3),
                        (4, 3),
                        (5, 3),
                        (6, 3),
                        (7, 3),
                        (8, 3),
                    ]
                )
            elif ranking == "elderly_first":
                return np.array(
                    [
                        (8, 2),
                        (7, 2),
                        (6, 2),
                        (5, 2),
                        (4, 2),
                        (3, 2),
                        (2, 2),
                        (1, 2),
                        (0, 2),
                        (8, 3),
                        (7, 3),
                        (6, 3),
                        (5, 3),
                        (4, 3),
                        (3, 3),
                        (2, 3),
                        (1, 3),
                        (0, 3),
                    ]
                )
            elif ranking == "risk_ranked":
                assert (
                    self.severity_factorisation is not None
                ), "Need to pass severity factorisation to compute risk ranked vaccination policy"
                return self.ranking_from_risk_factors(
                    self.severity_factorisation.g, how="desc"
                )
            elif ranking == "risk_ranked_reversed":
                assert (
                    self.severity_factorisation is not None
                ), "Need to pass severity factorisation to compute risk ranked vaccination policy"
                return self.ranking_from_risk_factors(
                    self.severity_factorisation.g, how="asc"
                )
            else:
                ValueError(
                    f"{ranking} is not a valid ranking option. Options: 'young_first', 'elderly_first', 'risk_ranked'"
                )
        elif isinstance(ranking, list):
            return np.array(ranking)
        else:
            TypeError(f"ranking has invalid type {type(ranking)}")

    def _assign_first_and_second_doses(self, U_2, unassigned_vaccinations, a, t1, t2):
        vaccination_potential = (
            self.vaccine_acceptance_rate[a, 1] * self.params.D_a[a] - U_2[a, ...].sum()
        )
        if (vaccination_potential <= 0) or (t2 - t1) < self.params.l:
            return U_2, unassigned_vaccinations
        else:
            # twice vaccinated
            available_first_doses = unassigned_vaccinations[t1, 0]
            available_second_doses = unassigned_vaccinations[t2, 1]
            assigned_first_and_second = min(
                available_first_doses, available_second_doses, vaccination_potential
            )
            U_2[a, t1, t2] = assigned_first_and_second

            unassigned_vaccinations[t1, 0] -= assigned_first_and_second
            unassigned_vaccinations[t2, 1] -= assigned_first_and_second
            return U_2, unassigned_vaccinations

    def _assign_third_doses(self, u_3, unassigned_vaccinations, U_2, a, t2, t3):
        U_2_repeat = np.repeat(
            U_2[a, :-1, :-1, np.newaxis], U_2.shape[1] - 1, axis=2
        )  # copy over dimension t3
        u_3_repeat = np.repeat(
            u_3[a, np.newaxis, :-1, :-1], u_3.shape[1] - 1, axis=0
        )  # copy over dimension t1
        already_assigned_third_doses = (U_2_repeat * u_3_repeat).sum()
        vaccination_potential = (
            self.vaccine_acceptance_rate[a, 2] * self.params.D_a[a]
            - already_assigned_third_doses
        )
        available_third_doses = unassigned_vaccinations[t3, 2]
        already_assigned_second_doses_relative = u_3[a, t2, :].sum()
        second_doses = U_2[a, :, t2].sum()
        available_second_doses = (
            1 - already_assigned_second_doses_relative
        ) * second_doses
        assigned_third_doses = min(
            available_second_doses, available_third_doses, vaccination_potential
        )

        if available_second_doses > 0:  # avoid division by 0
            u_3[a, t2, t3] = assigned_third_doses / second_doses
        else:
            u_3[a, t2, t3] = 0
        unassigned_vaccinations[t3, 2] -= assigned_third_doses
        return u_3, unassigned_vaccinations

    def _assign_first_only_doses(self, U_2, unassigned_vaccinations, a, t1):
        # only once vaccinated
        vaccination_potential = (
            self.vaccine_acceptance_rate[a, 0] * self.params.D_a[a] - U_2[a, ...].sum()
        )
        if vaccination_potential <= 0:
            return U_2, unassigned_vaccinations
        else:
            available_first_doses = unassigned_vaccinations[t1, 0]
            assigned_first_doses = min(available_first_doses, vaccination_potential)
            U_2[a, t1, -1] += assigned_first_doses
            unassigned_vaccinations[t1, 0] -= assigned_first_doses
            return U_2, unassigned_vaccinations

    def _assign_vaccinations(self, U_2, u_3, unassigned_vaccinations, t, a, v):
        if v == 2:
            t1 = t
            for gap in self.gap_ranking[
                t1 + self.gap_ranking <= self.params.weeks.max()
            ]:
                t2 = t1 + gap
                U_2, unassigned_vaccinations = self._assign_first_and_second_doses(
                    U_2,
                    unassigned_vaccinations,
                    a,
                    t1,
                    t2,
                )
        elif v == 3:
            t3 = t
            k = self.params.k
            for t2 in reversed(self.params.weeks[: max(0, t3 - k + 1)]):
                u_3, unassigned_vaccinations = self._assign_third_doses(
                    u_3, unassigned_vaccinations, U_2, a, t2, t3
                )
        return U_2, u_3, unassigned_vaccinations

    @staticmethod
    def ranking_from_risk_factors(g: npt.ArrayLike, how: str = "desc") -> npt.ArrayLike:
        global is_asc
        if how == "asc":
            is_asc = True
        elif how == "desc":
            is_asc = False
        else:
            ValueError()

        risk_factors_unvaccinated = g[0, :]
        risk_factors_unvaccinated_sorted = (
            np.argsort(risk_factors_unvaccinated)
            if is_asc
            else np.argsort(risk_factors_unvaccinated)[::-1]
        )

        # assumption: risk ranking does not change by vaccination status
        for a in range(1, g.shape[0]):
            other = np.argsort(g[0, :]) if is_asc else np.argsort(g[0, :])[::-1]
            assert np.all(
                risk_factors_unvaccinated_sorted == other
            ), "risk ranking depends on vaccination status"

        ranking = [(a, 2) for a in risk_factors_unvaccinated_sorted]
        ranking.extend([(a, 3) for a in risk_factors_unvaccinated_sorted])
        return np.array(ranking)
