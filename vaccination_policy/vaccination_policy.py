from __future__ import annotations

import logging
from itertools import product
from pathlib import Path

import numpy as np
import numpy.typing as npt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from causal_model import Parameters

logger = logging.getLogger(__name__)

EPS = 1e-7  # epsilon value used to check if values are bigger than zero


class VaccinationPolicy:
    subdir_name = "vaccination_policy"

    def __init__(self, params: Parameters, U_2: npt.NDArray, u_3: npt.NDArray) -> None:
        self.dims = (
            len(params.age_groups),
            len(params.weeks) + 1,
            len(params.weeks) + 1,
        )
        self.params = params
        self.U_2 = self._truncate_U_2(
            U_2, params.weeks_extended, params.weeks_scenario_extended
        )
        self.u_3 = self._truncate_u_3(
            u_3, params.weeks_extended, params.weeks_scenario_extended
        )
        self.U_2_full = U_2
        self.u_3_full = u_3
        self.D_a = params.D_a
        self.expected_vaccinations = params.observed_vaccinations[
            :, : len(params.weeks_scenario), :
        ]

    def validate(self) -> bool:
        """
        Function that checks all assumptions on vaccination policy:
        - normalisation of probabilities
        - gaps between doses
        - positive vaccination numbers
        """
        # normalisation
        np.testing.assert_array_almost_equal(
            self.U_2.sum(axis=(1, 2)),
            self.D_a,
            err_msg="U_2 is not properly normalised",
        )
        assert np.all(self.U_2 >= -EPS)
        np.testing.assert_array_almost_equal(
            self.u_3.sum(axis=2),
            1,
            err_msg="u_3 is not properly normalised",
        )
        assert np.all(
            self.u_3 >= 0 - 1e-10
        ), f"u_3 has negative values, min(u_3)={self.u_3.min()}"  # TODO: change to int
        assert np.all(
            self.u_3 <= 1 + 1e-10
        ), f"u_3 has values larger than 1, max(u_3)={self.u_3.max()}"  # TODO: change to int

        np.testing.assert_array_almost_equal(
            self.u_3[:, -1, :],
            1.0 / len(self.params.weeks_scenario_extended),
            err_msg="u_3 has wrong dummy values",
        )

        # number of doses
        total_doses_per_week_expected = self.expected_vaccinations.sum(axis=(0, 2))
        first_doses_policy = self.U_2.sum(axis=(0, 2))[:-1]
        second_doses_policy = self.U_2.sum(axis=(0, 1))[:-1]
        U_2_repeat = np.repeat(
            self.U_2[:, :-1, :-1, np.newaxis], self.U_2.shape[1] - 1, axis=3
        )  # copy over dimension t3
        u_3_repeat = np.repeat(
            self.u_3[:, np.newaxis, :-1, :-1], self.u_3.shape[1] - 1, axis=1
        )  # copy over dimension t1
        third_doses_policy = (U_2_repeat * u_3_repeat).sum(axis=(0, 1, 2))
        total_doses_per_week_policy = (
            first_doses_policy + second_doses_policy + third_doses_policy
        )
        np.testing.assert_array_almost_equal(
            total_doses_per_week_expected,
            total_doses_per_week_policy,
            err_msg="Doses in policy do not match expected doses",
        )

        # TODO: check gaps between doses
        for a, t1, t2 in product(
            self.params.age_groups,
            self.params.weeks_scenario,
            self.params.weeks_scenario,
        ):
            if t2 < t1 + self.params.l:
                assert self.U_2[a, t1, t2] == 0, (
                    f"Policy does not respect minimum distance"
                    f" between 1st and 2nd dose for [a, t1, t2]: {a, t1, t2}"
                )

        for a, t2, t3 in product(
            self.params.age_groups,
            self.params.weeks_scenario,
            self.params.weeks_scenario,
        ):
            if t3 < t2 + self.params.k:
                assert self.u_3[a, t2, t3] == 0, (
                    f"Policy does not respect minimum distance"
                    f" between 2nd and 3rd dose for [a, t2, t3] = {a, t2, t3},"
                    f" with u_3[{a}, {t2}, {t3}] = {self.u_3[a, t2, t3]}"
                )
        logger.info("Validation successful")

        return True

    @staticmethod
    def _truncate_U_2(U_2, weeks_extended, weeks_scenario_extended):
        t_slice_removed = slice(
            len(weeks_scenario_extended) - 1, len(weeks_extended) - 1
        )
        remove_element_mask = np.zeros_like(weeks_extended, dtype=bool)
        remove_element_mask[t_slice_removed] = True

        # delete along t1 axis
        # delete all 1st doses after scenario time window and move to unvaccinated
        U_2[..., -1, -1] += U_2[:, t_slice_removed, :].sum(axis=(1, 2))
        U_2 = np.delete(U_2, remove_element_mask, axis=1)

        # squash along t2 axis
        # move all not-yet materialised 2nd doses to unvaccinated
        U_2[..., -1] += U_2[..., t_slice_removed].sum(axis=2)
        U_2 = np.delete(U_2, remove_element_mask, axis=2)

        return U_2

    @staticmethod
    def _truncate_u_3(u_3, weeks_extended, weeks_scenario_extended):
        t_slice_removed = slice(
            len(weeks_scenario_extended) - 1, len(weeks_extended) - 1
        )
        remove_element_mask = np.zeros_like(weeks_extended, dtype=bool)
        remove_element_mask[t_slice_removed] = True

        # delete along t1 axis
        # delete second doses after scenario time window from the conditioning set
        u_3 = np.delete(u_3, remove_element_mask, axis=1)

        # squash along t3 axis
        # move all not-yet materialised 3rd doses to unboostered
        u_3[:, :-1, -1] += u_3[:, :-1, t_slice_removed].sum(axis=2)

        # adjust dummy values
        u_3[:, -1, :] = 1.0 / (len(weeks_scenario_extended))
        u_3 = np.delete(u_3, remove_element_mask, axis=2)

        return u_3

    def save(self, dir: Path) -> None:
        subdir = dir / self.subdir_name
        subdir.mkdir(parents=True, exist_ok=False)
        np.save(subdir / "U_2", self.U_2)
        np.save(subdir / "u_3", self.u_3)
        np.save(subdir / "U_2_full", self.U_2_full)
        np.save(subdir / "u_3_full", self.u_3_full)


class ObservedVaccinationPolicy(VaccinationPolicy):
    pass


class FixedDosesVaccinationPolicy(VaccinationPolicy):
    pass


class GeneralisedVaccinationPolicy(VaccinationPolicy):
    def __init__(
        self,
        params: Parameters,
        U_2: npt.NDArray,
        u_3: npt.NDArray,
        expected_vaccinations: npt.NDArray,
    ) -> None:
        super().__init__(params=params, U_2=U_2, u_3=u_3)
        self.expected_vaccinations = expected_vaccinations
