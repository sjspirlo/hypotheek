def annual_to_monthly(annual_rate: float) -> float:
    """Monthly rate that compounds to `annual_rate` over 12 months."""
    return (1 + annual_rate) ** (1 / 12) - 1
