from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from vaccination_policy import VaccinationPolicy

if TYPE_CHECKING:
    from causal_model import Parameters


class VaccinationPolicyGenerator(ABC):
    @abstractmethod
    def __init__(self, params: Parameters, **kwargs):
        self.params = params
        self.parametrisation_dims = (
            len(self.params.age_groups),
            len(self.params.weeks) + 1,
            len(self.params.weeks) + 1,
        )

    @abstractmethod
    def generate(self) -> VaccinationPolicy:
        pass
