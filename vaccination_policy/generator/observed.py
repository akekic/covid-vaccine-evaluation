from __future__ import annotations

from collections import Iterable
from typing import TYPE_CHECKING, Union

import numpy as np

from vaccination_policy import (
    VaccinationPolicy,
    ObservedVaccinationPolicy,
    GeneralisedVaccinationPolicy,
)
from .base import VaccinationPolicyGenerator


if TYPE_CHECKING:
    from causal_model import Parameters


class ObservedVaccinationPolicyGenerator(VaccinationPolicyGenerator):
    def __init__(
        self,
        params: Parameters,
        ppt_acceptance_change: Union[float, Iterable[float]] = 0.0,
    ):
        super().__init__(params)
        if isinstance(ppt_acceptance_change, float):
            self.ppt_acceptance_change = ppt_acceptance_change * np.ones_like(
                params.age_groups
            )
        elif isinstance(ppt_acceptance_change, Iterable):
            self.ppt_acceptance_change = np.array(ppt_acceptance_change)
        else:
            raise TypeError(
                f"Type {type(ppt_acceptance_change)} not supported for ppt_acceptance_change"
            )

    def generate(self) -> VaccinationPolicy:
        """
        Simplifying assumptions:
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
        unassigned_vaccinations = self._prepare_unassigned_vaccinations()
        gap_ranking = np.arange(l, 40)
        for a in self.params.age_groups:
            for t2 in self.params.weeks[min(gap_ranking) :]:
                for gap in gap_ranking[gap_ranking <= t2]:
                    second_vaccinations = unassigned_vaccinations[a, t2, 1]
                    first_vaccinations = unassigned_vaccinations[a, t2 - gap, 0]
                    assigned_first_and_second = min(
                        first_vaccinations, second_vaccinations
                    )
                    # twice vaccinated
                    U_2[a, t2 - gap, t2] += assigned_first_and_second

                    unassigned_vaccinations[a, t2 - gap, 0] -= assigned_first_and_second
                    unassigned_vaccinations[a, t2, 1] -= assigned_first_and_second
            for t1 in self.params.weeks[:-l]:
                # only once vaccinated
                assigned_first_only = unassigned_vaccinations[a, t1, 0]
                U_2[a, t1, -1] += assigned_first_only
                unassigned_vaccinations[a, t1, 0] -= assigned_first_only

            for t1 in self.params.weeks[-l:]:
                # only once vaccinated (no time for second shot)
                assigned_first_only = unassigned_vaccinations[a, t1, 0]
                U_2[a, t1, -1] += assigned_first_only
                unassigned_vaccinations[a, t1, 0] -= assigned_first_only

            # completely unvaccinated
            U_2[a, -1, -1] = D_a[a] - U_2[a, ...].sum()

            for t3 in self.params.weeks:
                denominator = U_2[a, :, 1 : t3 - k].sum()
                if denominator == 0:  # avoid division by 0
                    u_3[a, :, t3] = 0
                else:
                    u_3[a, 1 : t3 - k, t3] = (
                        unassigned_vaccinations[a, t3, 2] / denominator
                    )

            for t2 in self.params.weeks:
                # only twice vaccinated
                u_3[a, t2, -1] = 1 - u_3[a, t2, :].sum()

        # these are dummy values which make the summation in the target function simpler
        # it does not mean that there are peole who got their 3rd but not their 2nd shot
        u_3[:, -1, :] = 1.0 / (len(self.params.weeks) + 1)

        if (self.ppt_acceptance_change == 0).all():
            policy = ObservedVaccinationPolicy(self.params, U_2=U_2, u_3=u_3)
        else:
            policy = GeneralisedVaccinationPolicy(
                self.params,
                U_2=U_2,
                u_3=u_3,
                expected_vaccinations=self._prepare_unassigned_vaccinations(),
            )
        # policy.validate()  # TODO: why is this commented out?
        return policy

    def _prepare_unassigned_vaccinations(self):
        # scale up or down according to ppt acceptance change
        post_intervention_vaccine_acceptance_rate = (
            self.params.vaccine_acceptance_rate.T + 0.01 * self.ppt_acceptance_change
        ).T
        if (post_intervention_vaccine_acceptance_rate > 1.0).any():
            raise ValueError(
                "Encountered post intervention vaccine acceptance rates bigger than 100%"
            )
        scaling_factors = (
            post_intervention_vaccine_acceptance_rate
            / self.params.vaccine_acceptance_rate
        )
        return scaling_factors[:, np.newaxis, :] * np.copy(
            self.params.observed_vaccinations
        )
