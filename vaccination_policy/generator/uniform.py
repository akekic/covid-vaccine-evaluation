from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vaccination_policy import VaccinationPolicy, FixedDosesVaccinationPolicy
from .base import VaccinationPolicyGenerator

if TYPE_CHECKING:
    from causal_model import Parameters


class UniformVaccinationPolicyGenerator(VaccinationPolicyGenerator):
    def __init__(self, params: Parameters, use_vaccine_acceptance: bool = False):
        super().__init__(params)
        self.use_vaccine_acceptance = use_vaccine_acceptance
        if use_vaccine_acceptance:
            var = params.vaccine_acceptance_rate.copy()
            a = np.zeros_like(var)
            a[:, 0] = var[:, 0] - var[:, 1]
            a[:, 1] = var[:, 1]
            a[:, 2] = var[:, 2] + np.clip(
                0.07 * np.ones_like(var[:, 1]), 0, var[:, 1] - var[:, 2]
            )
            a = np.stack((params.P_a, params.P_a, params.P_a)).T * a
            self.dose_distribution = a / a.sum(axis=0)
        else:
            self.dose_distribution = np.stack((params.P_a, params.P_a, params.P_a)).T

    def generate(self) -> VaccinationPolicy:
        """
        Simplifying assumptions:
        - vaccination probability same across all age groups
        - t2 at least l weeks after t1
        - t3 at least k weeks after t2
        - vaccination policy has same number of observed vaccinations (1st, 2nd and 3rd dose)
        """

        U_2, u_3 = np.zeros(self.parametrisation_dims), np.zeros(
            self.parametrisation_dims
        )

        l = self.params.l
        k = self.params.k
        D_a = self.params.D_a

        unassigned_vaccinations = np.copy(self.params.observed_vaccinations).sum(axis=0)
        gap_ranking = np.arange(l, 40)
        for t2 in self.params.weeks[min(gap_ranking) :]:
            for gap in gap_ranking[gap_ranking <= t2]:
                second_vaccinations = unassigned_vaccinations[t2, 1]
                first_vaccinations = unassigned_vaccinations[t2 - gap, 0]
                assigned_first_and_second = min(first_vaccinations, second_vaccinations)
                # twice vaccinated
                U_2[:, t2 - gap, t2] += (
                    self.dose_distribution[:, 1] * assigned_first_and_second
                )

                unassigned_vaccinations[t2 - gap, 0] -= assigned_first_and_second
                unassigned_vaccinations[t2, 1] -= assigned_first_and_second

        for t1 in self.params.weeks[:-l]:
            # only once vaccinated
            assigned_first_only = unassigned_vaccinations[t1, 0]
            U_2[:, t1, -1] += self.dose_distribution[:, 0] * assigned_first_only
            unassigned_vaccinations[t1, 0] -= assigned_first_only

        for t1 in self.params.weeks[-l:]:
            # only once vaccinated (no time for second shot)
            assigned_first_only = unassigned_vaccinations[t1, 0]
            U_2[:, t1, -1] += self.dose_distribution[:, 0] * assigned_first_only
            unassigned_vaccinations[t1, 0] -= assigned_first_only

        # completely unvaccinated
        U_2[:, -1, -1] = D_a - U_2.sum(axis=(1, 2))

        for t3 in self.params.weeks:
            denominator = U_2[:, :, l : t3 - k].sum(axis=(1, 2))
            if denominator.sum() == 0:  # avoid division by 0
                u_3[:, :, t3] = 0
            else:
                u_3[:, 1 : t3 - k, t3] = (
                    self.dose_distribution[:, 2]
                    * unassigned_vaccinations[t3, 2]
                    / denominator
                ).reshape((-1, 1))

        for t2 in self.params.weeks:
            # only twice vaccinated
            u_3[:, t2, -1] = 1 - u_3[:, t2, :].sum(axis=-1)

        # these are dummy values which make the summation in the target function simpler
        # it does not mean that there are people who got their 3rd but not their 2nd shot
        u_3[:, -1, :] = 1.0 / (len(self.params.weeks) + 1)

        policy = FixedDosesVaccinationPolicy(self.params, U_2=U_2, u_3=u_3)
        policy.validate()
        return policy
