from abc import ABC, abstractmethod

import numpy as np

from mortgage_simulator.custom_types import FloatArray, IntArray


class Mortgage(ABC):
    """
    Base class for a mortgage with monthly payments.

    Parameters
    ----------
    principal : float
        Loan amount.
    annual_rate : float
        Nominal yearly interest rate as a fraction, e.g. 0.04 for 4%.
    maturity : int
        Duration of the mortgage in months.
    """

    def __init__(
        self,
        principal: float,
        annual_rate: float,
        maturity: int,
    ) -> None:
        if principal <= 0:
            raise ValueError("principal must be positive")
        if annual_rate < 0:
            raise ValueError("annual_rate cannot be negative")
        if maturity <= 0:
            raise ValueError("maturity must be positive")

        self.principal = principal
        self.annual_rate = annual_rate
        self.maturity = maturity

    @property
    def payment_ts(self) -> IntArray:
        return np.arange(self.maturity)

    @property
    def monthly_rate(self) -> float:
        "The bank actually charges annual_rate / 12, not (1 + annual_rate)^(1/12) -1."
        return self.annual_rate / 12

    @property
    def maturity_in_years(self) -> float:
        return self.maturity / 12

    @abstractmethod
    def outstanding_principal(
        self,
        t: int,
    ) -> float: ...

    def gross_interest_payment(
        self,
        t: int,
    ) -> float:
        return self.monthly_rate * self.outstanding_principal(t=t)

    def principal_payment_t(
        self,
        t: int,
    ) -> float:

        return self.outstanding_principal(
            t=t,
        ) - self.outstanding_principal(
            t=t + 1,
        )

    def gross_interest_payment_t(
        self,
        t: int,
    ) -> float:
        return self.outstanding_principal(t=t) * self.monthly_rate

    def total_gross_payment_t(
        self,
        t: int,
    ) -> float:
        return self.gross_interest_payment_t(t=t) + self.principal_payment_t(t=t)

    @property
    def amortization_scheme(self) -> FloatArray:
        amor_scheme = np.zeros(self.maturity)
        for t in range(self.maturity):
            amor_scheme[t] = self.outstanding_principal(t=t)
        return amor_scheme

    @property
    def principal_payment_scheme(self) -> FloatArray:
        principal_scheme = np.zeros(self.maturity)
        for t in range(self.maturity):
            principal_scheme[t] = self.principal_payment_t(t=t)
        return principal_scheme

    @property
    def gross_interest_payment_scheme(self) -> FloatArray:
        gross_interest_scheme = np.zeros(self.maturity)
        for t in range(self.maturity):
            gross_interest_scheme[t] = self.gross_interest_payment_t(t=t)
        return gross_interest_scheme

    @property
    def total_gross_payment_scheme(self) -> FloatArray:
        return self.gross_interest_payment_scheme + self.principal_payment_scheme

    def equity_built_t(
        self,
        t: int,
    ) -> float:
        return self.principal - self.outstanding_principal(t=t)

    def fractional_equity_built_t(
        self,
        t: int,
    ) -> float:
        return self.equity_built_t(t=t) / self.principal

    @property
    def equity_scheme(self) -> FloatArray:
        equity_scheme = np.zeros(self.maturity)
        for t in range(self.maturity):
            equity_scheme[t] = self.equity_built_t(t=t)
        return equity_scheme

    @property
    def fractional_equity_scheme(self) -> FloatArray:
        return self.equity_scheme / self.principal


class Annuity(Mortgage):
    """Constant monthly payment; the share of repayment grows over time."""

    def outstanding_principal(
        self,
        t: int,
    ) -> float:
        num = (
            (1 + self.monthly_rate) ** t - (1 + self.monthly_rate) ** self.maturity
        ) * self.principal
        denom = 1 - (1 + self.monthly_rate) ** self.maturity
        return num / denom


class Linear(Mortgage):
    """Equal principal repayment every month; payments decrease over time."""

    def outstanding_principal(
        self,
        t: int,
    ) -> float:
        return self.principal * (t / self.maturity)


class Bullet(Mortgage):
    """Interest only; the full principal is repaid in the last month."""

    def outstanding_principal(
        self,
        t: int,
    ) -> float:
        return self.principal
