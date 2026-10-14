from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mortgage_simulator.custom_types import FloatArray
from mortgage_simulator.mortgages import Mortgage
from mortgage_simulator.rates import RatePath, growth_factors, monthly_rates


@dataclass(frozen=True)
class HousingMarket:
    """
    Expected development of the house price.

    Parameters
    ----------
    growth : RatePath
        Monthly house price growth.
    """

    growth: RatePath


@dataclass(frozen=True)
class SavingsPlan:
    """
    How much money is available each month and how savings grow.

    Parameters
    ----------
    initial_savings : float
        Savings at the start of the simulation, after paying the down payment
        and purchase costs.
    monthly_budget : float
        Money available in the first month for housing and saving, i.e. net
        income minus living expenses. The mortgage payment and housing costs are
        paid from this; whatever is left is saved.
    budget_growth : RatePath | None
        Monthly growth of the budget, e.g. salary increases. None means no growth.
    savings_return : RatePath | None
        Monthly return on savings. None means no return.
    """

    initial_savings: float
    monthly_budget: float
    budget_growth: RatePath | None = None
    savings_return: RatePath | None = None


@dataclass(frozen=True)
class RecurringCost:
    """
    A monthly housing cost such as VvE contributions or insurance.

    Parameters
    ----------
    name : str
        Label used in reports.
    monthly_amount : float
        Cost in the first month.
    increase : RatePath | None
        Monthly increase of the cost, e.g. indexation with inflation. None
        means the cost stays the same.
    """

    name: str
    monthly_amount: float
    increase: RatePath | None = None


@dataclass(frozen=True)
class FinancialReport:
    """Snapshot of the financial position after `month` months."""

    month: int
    house_value: float
    mortgage_balance: float
    home_equity: float
    savings: float
    net_worth: float
    cumulative_interest_paid: float
    cumulative_housing_costs: float

    @property
    def years(self) -> float:
        return self.month / 12


class FinancialSimulation:
    """
    Month-by-month simulation of a mortgage, the house value and savings.

    Month 0 is the purchase date. In month k (1..horizon) the k-th mortgage
    payment and the housing costs are paid from the budget, savings earn the
    return of month k, and the remainder of the budget is added to savings.
    After the mortgage matures its payment is zero.

    Every RatePath must cover at least `horizon` months; longer paths are cut off.

    Parameters
    ----------
    mortgage : Mortgage
        The mortgage financing the house.
    house_price : float
        Value of the house at purchase.
    housing_market : HousingMarket
        House price growth assumptions.
    savings_plan : SavingsPlan
        Budget and savings assumptions.
    costs : Sequence[RecurringCost]
        Other monthly housing costs (VvE, insurances, ...).
    horizon : int | None
        Number of months to simulate; defaults to the mortgage maturity.
    """

    def __init__(
        self,
        mortgage: Mortgage,
        house_price: float,
        housing_market: HousingMarket,
        savings_plan: SavingsPlan,
        costs: Sequence[RecurringCost] = (),
        horizon: int | None = None,
    ) -> None:
        if house_price <= 0:
            raise ValueError("house_price must be positive")
        if horizon is not None and horizon <= 0:
            raise ValueError("horizon must be positive")

        self.mortgage = mortgage
        self.house_price = house_price
        self.housing_market = housing_market
        self.savings_plan = savings_plan
        self.costs = tuple(costs)
        self.horizon = mortgage.maturity if horizon is None else horizon

        paths: dict[str, RatePath | None] = {
            "housing_market.growth": housing_market.growth,
            "savings_plan.budget_growth": savings_plan.budget_growth,
            "savings_plan.savings_return": savings_plan.savings_return,
        }
        paths.update({f"cost '{cost.name}'.increase": cost.increase for cost in costs})
        for name, path in paths.items():
            if path is not None and path.months < self.horizon:
                raise ValueError(
                    f"{name} covers {path.months} months, "
                    f"but the horizon is {self.horizon} months",
                )

    def mortgage_balance_t(
        self,
        t: int,  # months after purchase
    ) -> float:
        return self.mortgage.outstanding_principal(t=min(t, self.mortgage.maturity))

    def mortgage_payment_k(
        self,
        k: int,  # 1-based month in which the payment is made
    ) -> tuple[float, float]:
        """Interest and principal paid in month k."""
        if k > self.mortgage.maturity:
            return 0.0, 0.0
        return (
            self.mortgage.gross_interest_payment_t(t=k - 1),
            self.mortgage.principal_payment_t(t=k - 1),
        )

    def house_values(self) -> FloatArray:
        """House value at months 0..horizon."""
        return self.house_price * growth_factors(
            path=self.housing_market.growth,
            months=self.horizon,
        )

    def budgets(self) -> FloatArray:
        """Budget in months 0..horizon; month 0 has no budget."""
        plan = self.savings_plan
        factors = growth_factors(path=plan.budget_growth, months=self.horizon)
        return np.concatenate([[0.0], plan.monthly_budget * factors[:-1]])

    def housing_costs(self) -> FloatArray:
        """Total recurring costs in months 0..horizon; month 0 has no costs."""
        total = np.zeros(self.horizon + 1)
        for cost in self.costs:
            factors = growth_factors(path=cost.increase, months=self.horizon)
            total[1:] += cost.monthly_amount * factors[:-1]
        return total

    def simulate(self) -> pd.DataFrame:
        """
        Full monthly path, indexed by month 0..horizon.

        Flow columns (payments, costs, saved) hold the amounts of that month and
        are zero in month 0. Stock columns (balances, values) are the positions
        at the end of that month.
        """
        months = np.arange(self.horizon + 1)
        interest: FloatArray = np.zeros(self.horizon + 1)
        principal: FloatArray = np.zeros(self.horizon + 1)
        for k in range(1, self.horizon + 1):
            interest[k], principal[k] = self.mortgage_payment_k(k=k)

        housing_costs = self.housing_costs()
        saved = self.budgets() - interest - principal - housing_costs

        returns = monthly_rates(
            path=self.savings_plan.savings_return,
            months=self.horizon,
        )
        savings: FloatArray = np.zeros(self.horizon + 1)
        savings[0] = self.savings_plan.initial_savings
        for k in range(1, self.horizon + 1):
            savings[k] = savings[k - 1] * (1 + returns[k - 1]) + saved[k]

        house_value = self.house_values()
        mortgage_balance = np.array([self.mortgage_balance_t(t=int(t)) for t in months])
        home_equity = house_value - mortgage_balance

        return pd.DataFrame(
            {
                "house_value": house_value,
                "mortgage_balance": mortgage_balance,
                "home_equity": home_equity,
                "savings": savings,
                "net_worth": home_equity + savings,
                "interest_paid": interest,
                "principal_paid": principal,
                "mortgage_payment": interest + principal,
                "housing_costs": housing_costs,
                "saved": saved,
                "cumulative_interest_paid": np.cumsum(interest),
                "cumulative_housing_costs": np.cumsum(housing_costs),
            },
            index=pd.Index(months, name="month"),
        )

    def report(
        self,
        month: int,  # months after purchase
    ) -> FinancialReport:
        if not 0 <= month <= self.horizon:
            raise ValueError(f"month must be between 0 and {self.horizon}")

        row = self.simulate().iloc[month]
        return FinancialReport(
            month=month,
            house_value=float(row["house_value"]),
            mortgage_balance=float(row["mortgage_balance"]),
            home_equity=float(row["home_equity"]),
            savings=float(row["savings"]),
            net_worth=float(row["net_worth"]),
            cumulative_interest_paid=float(row["cumulative_interest_paid"]),
            cumulative_housing_costs=float(row["cumulative_housing_costs"]),
        )
