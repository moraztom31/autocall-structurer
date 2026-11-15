"""
Deal definition and observation schedule construction.

A Phoenix autocallable is fully described by its barriers, its coupon,
and the dates on which those are tested. This module turns a set of
terms into the arrays the pricer consumes: observation times, discount
factors, and a flag marking which dates carry an autocall test.
"""

from dataclasses import dataclass

import numpy as np

NOMINAL = 1000.0


@dataclass(frozen=True)
class Deal:
    """
    Terms of a Phoenix autocallable note.

    Barriers are expressed as fractions of the initial index level.
    A Phoenix keeps the coupon barrier and the autocall barrier separate;
    setting them equal and accruing coupons would make it an Athena.
    """

    maturity_years: float = 5.0
    coupon_rate_pa: float = 0.06        # headline annual coupon
    coupons_per_year: int = 4           # quarterly observation
    coupon_barrier: float = 0.60
    autocall_barrier: float = 1.00
    autocall_start_year: float = 1.0    # first call date, in years
    autocalls_per_year: int = 1         # annual call schedule
    capital_barrier: float = 0.60
    issue_price: float = 1.00
    margin_floor: float = 0.015

    @property
    def coupon_amount(self):
        """Cash coupon per observation date, in currency per note."""
        return self.coupon_rate_pa / self.coupons_per_year * NOMINAL

    @property
    def n_observations(self):
        return int(round(self.maturity_years * self.coupons_per_year))

    def describe(self):
        """One-line summary used in headers and chart captions."""
        return (
            f"{self.maturity_years:g}Y Phoenix - "
            f"{self.coupon_rate_pa:.2%} p.a. - "
            f"{self.coupon_barrier:.0%} coupon barrier - "
            f"autocall at {self.autocall_barrier:.0%}"
        )


@dataclass(frozen=True)
class Schedule:
    """
    Observation grid for a deal.

    times        year fractions from the trade date
    df_ois       risk-free discount factor at each time
    is_autocall  True where the date carries an autocall test
    labels       display labels, either real dates or year fractions
    """

    times: np.ndarray
    df_ois: np.ndarray
    is_autocall: np.ndarray
    labels: list

    def __len__(self):
        return len(self.times)

    def forward_rates(self):
        """
        Continuously compounded forwards implied by the discount factors.

        These drive the index drift: the hedge is funded at the risk-free
        rate, so the issuer credit spread never enters here.
        """
        df_prev = np.concatenate([[1.0], self.df_ois[:-1]])
        t_prev = np.concatenate([[0.0], self.times[:-1]])
        dt = self.times - t_prev
        return np.log(df_prev / self.df_ois) / dt, dt

    def df_note(self, credit_spread):
        """
        Discount factors for the note's own cash flows.

            DF_note(T) = DF_ois(T) * exp(-s * T)

        The note is an unsecured obligation of the issuer, so its flows
        discount at the risk-free rate plus a credit spread.
        """
        return self.df_ois * np.exp(-credit_spread * self.times)


def build_schedule(deal, flat_rate, start_date=None):
    """
    Build an observation grid from deal terms and a flat rate.

    Used by the note builder, where no market discount curve is supplied.
    The flat rate is continuously compounded, so DF = exp(-r*T).

    A date carries an autocall test when it falls on the autocall grid
    at or after the first call date.
    """
    n = deal.n_observations
    step = 1.0 / deal.coupons_per_year
    times = np.array([(i + 1) * step for i in range(n)])

    call_step = 1.0 / deal.autocalls_per_year
    is_autocall = np.array([
        t >= deal.autocall_start_year - 1e-9
        and abs((t - deal.autocall_start_year) / call_step
                - round((t - deal.autocall_start_year) / call_step)) < 1e-6
        and i < n - 1                     # final date redeems, never calls
        for i, t in enumerate(times)
    ])

    df_ois = np.exp(-flat_rate * times)

    if start_date is not None:
        labels = [
            (start_date + np.timedelta64(int(round(t * 365.25)), "D"))
            .astype("datetime64[D]").astype(str)
            for t in times
        ]
    else:
        labels = [f"{t:.2f}y" for t in times]

    return Schedule(times=times, df_ois=df_ois,
                    is_autocall=is_autocall, labels=labels)


def schedule_from_table(rows):
    """
    Build a grid from an explicit table of (label, T, DF, is_autocall).

    Used by the worked example, where the discount curve is locked and
    must be reproduced exactly rather than approximated by a flat rate.
    """
    return Schedule(
        times=np.array([r[1] for r in rows]),
        df_ois=np.array([r[2] for r in rows]),
        is_autocall=np.array([r[3] for r in rows]),
        labels=[r[0] for r in rows],
    )
