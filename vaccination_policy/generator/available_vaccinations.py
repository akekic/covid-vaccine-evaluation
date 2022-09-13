from numpy import typing as npt


class UnassignedVaccinations:
    def __init__(
        self,
        initially_available_vaccinations: npt.ArrayLike,
        constraints_per_dose: bool,
    ):
        self.available_vaccinations = (
            initially_available_vaccinations.sum(axis=0)
            if constraints_per_dose
            else initially_available_vaccinations.sum(axis=(0, 2))
        )  # [t, dose]  or  [t]
        self.constraints_per_dose = constraints_per_dose

    def __getitem__(self, key):
        if isinstance(key, int):
            raise IndexError()
        elif isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError()
            else:
                if self.constraints_per_dose:
                    return self.available_vaccinations[key]
                else:
                    return self.available_vaccinations[key[0]]

    def __setitem__(self, key, value):
        if isinstance(key, int):
            raise IndexError()
        elif isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError()
            else:
                if self.constraints_per_dose:
                    self.available_vaccinations[key] = value
                else:
                    self.available_vaccinations[key[0]] = value

    def __repr__(self):
        return repr(self.available_vaccinations)
