"""
Worked example: a single deal priced on a frozen market snapshot.

Everything here is locked. A pricing committee prices the terms on the
reference snapshot agreed with trading, not on a live feed - otherwise
the numbers presented and the numbers approved are never the same.
Reproducibility is the point.

The only judgement left is the credit spread, picked from the ALM grid
and justified by the note's expected life.
"""

from deal import Deal, schedule_from_table

# ---------------------------------------------------------------------
# Locked snapshot
# ---------------------------------------------------------------------
TRADE_DATE = "2026-11-27"
SPOT = 2318.47          # index closing level on the initial fixing date
VOL = 0.17              # desk mid implied volatility, flat Black
DECREMENT = 0.05        # 5 % p.a. continuous, built into the published index

INDEX_NAME = "Euro healthcare index, 5 % decrement"

DEAL = Deal(
    maturity_years=5.0,
    coupon_rate_pa=0.06,
    coupons_per_year=4,
    coupon_barrier=0.60,
    autocall_barrier=1.00,
    autocall_start_year=1.0,
    autocalls_per_year=1,
    capital_barrier=0.60,
    issue_price=1.00,
    margin_floor=0.015,
)

# ---------------------------------------------------------------------
# Observation schedule with locked discount factors
# (label, T in years ACT/365.25, risk-free DF, autocall date?)
# ---------------------------------------------------------------------
TABLE = [
    ("2027-03-15", 0.2957, 0.99305, False),
    ("2027-06-15", 0.5476, 0.98619, False),
    ("2027-09-15", 0.7995, 0.97857, False),
    ("2027-12-15", 1.0486, 0.97035, True),
    ("2028-03-15", 1.2977, 0.96319, False),
    ("2028-06-15", 1.5496, 0.95591, False),
    ("2028-09-15", 1.8015, 0.94841, False),
    ("2028-12-15", 2.0507, 0.94111, True),
    ("2029-03-15", 2.2971, 0.93405, False),
    ("2029-06-15", 2.5489, 0.92709, False),
    ("2029-09-15", 2.8008, 0.91992, False),
    ("2029-12-15", 3.0500, 0.91284, True),
    ("2030-03-15", 3.2964, 0.90614, False),
    ("2030-06-15", 3.5483, 0.89902, False),
    ("2030-09-15", 3.8001, 0.89225, False),
    ("2030-12-15", 4.0493, 0.88525, True),
    ("2031-03-15", 4.2957, 0.87871, False),
    ("2031-06-15", 4.5476, 0.87168, False),
    ("2031-09-15", 4.7995, 0.86507, False),
    ("2031-12-15", 5.0486, 0.85815, False),   # final valuation
]

SCHEDULE = schedule_from_table(TABLE)

# ---------------------------------------------------------------------
# ALM credit grid: issuer funding spread over the risk-free curve
# ---------------------------------------------------------------------
CREDIT_GRID = {
    1: 0.0010,
    2: 0.0015,
    3: 0.0020,
    5: 0.0030,
}

CREDIT_RATIONALE = {
    1: "First autocall likely; the note exits at the first call date.",
    2: "Exit expected around year two, or holders rolled into a new theme.",
    3: "The note more likely than not survives the early call dates.",
    5: "Conservative: funded as if held to scheduled maturity.",
}


def nearest_grid_tenor(expected_life):
    """
    Grid line closest to a simulated expected life.

    Choosing the spread from the model rather than by eye matters: a
    wider spread lowers fair value and flatters the margin, so the choice
    has to be defensible on its own terms.
    """
    return min(CREDIT_GRID, key=lambda t: abs(t - expected_life))
