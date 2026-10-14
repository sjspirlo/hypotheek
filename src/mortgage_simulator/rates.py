from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from mortgage_simulator.custom_types import FloatArray
from mortgage_simulator.utils import annual_to_monthly


@dataclass(frozen=True, eq=False)
class RatePath:
    """
    Path of monthly growth rates; `rates[k - 1]` applies in month k.

    Use the constructors `constant`, `from_monthly` and `from_annual` rather than
    passing `rates` directly.

    Parameters
    ----------
    rates : FloatArray
        One-dimensional array of monthly rates as fractions, e.g. 0.0025.
    """

    rates: FloatArray

    def __post_init__(self) -> None:
        rates = np.array(self.rates, dtype=float)
        if rates.ndim != 1:
            raise ValueError("rates must be one-dimensional")
        if np.any(rates <= -1):
            raise ValueError("rates must be greater than -1")
        rates.setflags(write=False)
        object.__setattr__(self, "rates", rates)

    @classmethod
    def constant(
        cls,
        annual_rate: float,
        months: int,
    ) -> "RatePath":
        """The same yearly rate in every month."""
        return cls(np.full(months, annual_to_monthly(annual_rate)))

    @classmethod
    def zero(
        cls,
        months: int,
    ) -> "RatePath":
        return cls(np.zeros(months))

    @classmethod
    def from_monthly(
        cls,
        rates: npt.ArrayLike,
    ) -> "RatePath":
        """Monthly rates as given."""
        return cls(np.asarray(rates, dtype=float))

    @classmethod
    def from_annual(
        cls,
        annual_rates: npt.ArrayLike,
    ) -> "RatePath":
        """One yearly rate per year, held for the 12 months of that year."""
        monthly = (1 + np.asarray(annual_rates, dtype=float)) ** (1 / 12) - 1
        return cls(np.repeat(monthly, 12))

    @property
    def months(self) -> int:
        return len(self.rates)

    def growth_factors(self) -> FloatArray:
        """Cumulative growth since month 0; length months + 1, starting at 1."""
        return np.concatenate([[1.0], np.cumprod(1 + self.rates)])


def monthly_rates(
    path: RatePath | None,
    months: int,
) -> FloatArray:
    """Rates for months 1..months; zero when `path` is None."""
    if path is None:
        return np.zeros(months)
    return path.rates[:months]


def growth_factors(
    path: RatePath | None,
    months: int,
) -> FloatArray:
    """Cumulative growth for months 0..months; no growth when `path` is None."""
    if path is None:
        return np.ones(months + 1)
    return path.growth_factors()[: months + 1]
