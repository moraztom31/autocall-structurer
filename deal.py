"""
Deal definition and observation schedule construction.

An autocallable note is described by its barriers, its coupon mechanics,
and the dates on which those are tested. This module turns a set of
terms into the arrays the pricer consumes: observation times, discount
factors, a flag marking autocall dates, and the trigger level at each of
them.

Four features beyond the plain structure, each common in European retail
issuance:

  memory coupons      missed coupons are recovered when the barrier is
                      next met, rather than lost
  step-down trigger   the autocall level falls each year, so the note
                      becomes progressively easier to call
  Athena style        a single barrier, with coupons accrued and paid on
                      redemption rather than as they fall due
  airbag              below the barrier, loss is measured from a lower
                      strike, which cushions the first part of the fall
"""

from dataclasses import dataclass, replace

import numpy as np

NOMINAL = 1000.0

PHOENIX = "phoenix"
ATHENA = "athena"


@dataclass(frozen=True)
class Deal:
    """
    Terms of an autocallable note.

    Barriers are fractions of the initial index level.

    A Phoenix keeps the coupon barrier below the autocall barrier and
    pays coupons as they fall due. An Athena aligns them and accrues:
    nothing is paid until the note redeems, and then everything is.
    """

    style: str = PHOENIX

    maturity_years: float = 5.0
    coupon_rate_pa: float = 0.06
    coupons_per_year: int = 4
    coupon_barrier: float = 0.60
    coupon_memory: bool = False

    autocall_barrier: float = 1.00
    autocall_start_year: float = 1.0
    autocalls_per_year: int = 1
    autocall_stepdown_pa: float = 0.0     # trigger reduction per year

    capital_barrier: float = 0.60
    airbag: bool = False
    airbag_strike: float = 0.60           # loss measured from this level

    issue_price: float = 1.00
    margin_floor: float = 0.015

    def __post_init__(self):
        if self.style not in (PHOENIX, ATHENA):
            raise ValueError(f"unknown style: {self.style}")

    # -- derived ------------------------------------------------------
    @property
    def is_athena(self):
        return self.style == ATHENA

    @property
    def coupon_amount(self):
        """Cash coupon per observation date, per note."""
        return self.coupon_rate_pa / self.coupons_per_year * NOMINAL

    @property
    def n_observations(self):
        return int(round(self.maturity_years * self.coupons_per_year))

    @property
    def effective_coupon_barrier(self):
        """
        Level the coupon condition is tested against.

        An Athena has one barrier, so its coupon condition is the
        autocall level, not a separate lower one.
        """
        return (self.autocall_barrier if self.is_athena
                else self.coupon_barrier)

    def describe(self):
        """One-line summary for headers and captions."""
        bits = [f"{self.maturity_years:g}Y",
                "Athena" if self.is_athena else "Phoenix",
                f"{self.coupon_rate_pa:.2%} p.a."]
        if not self.is_athena:
            bits.append(f"{self.coupon_barrier:.0%} coupon barrier")
            if self.coupon_memory:
                bits.append("with memory")
        bits.append(f"autocall from {self.autocall_barrier:.0%}")
        if self.autocall_stepdown_pa:
            bits.append(f"stepping down {self.autocall_stepdown_pa:.1%} p.a.")
        if self.airbag:
            bits.append(f"airbag at {self.airbag_strike:.0%}")
        return " · ".join(bits)

    def features(self):
        """Active non-default features, for display as tags."""
        out = []
        if self.is_athena:
            out.append("Athena · coupons accrue to redemption")
        if self.coupon_memory and not self.is_athena:
            out.append("Memory coupons")
        if self.autocall_stepdown_pa:
            out.append(f"Step-down {self.autocall_stepdown_pa:.1%} p.a.")
        if self.airbag:
            out.append(f"Airbag from {self.airbag_strike:.0%}")
        return out

    def with_(self, **changes):
        """Copy with some terms replaced - used for lever testing."""
        return replace(self, **changes)


@dataclass(frozen=True)
class Schedule:
    """
    Observation grid.

    times            year fractions from the trade date
    df_ois           risk-free discount factor at each time
    is_autocall      True where the date carries an autocall test
    autocall_level   trigger as a fraction of initial, per date; already
                     reflects any step-down
    labels           display labels
    """

    times: np.ndarray
    df_ois: np.ndarray
    is_autocall: np.ndarray
    autocall_level: np.ndarray
    labels: list

    def __len__(self):
        return len(self.times)

    def forward_rates(self):
        """
        Continuously compounded forwards implied by the discount factors.

        These drive the index drift. The hedge is funded at the risk-free
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

    def call_levels(self):
        """(times, levels) for the autocall dates only."""
        mask = self.is_autocall
        return self.times[mask], self.autocall_level[mask]


def _autocall_levels(deal, times, is_autocall):
    """
    Trigger level at each date, after any step-down.

    The step-down accrues from the first call date, so the first call is
    always at the headline level and each later one is lower. The trigger
    is floored at zero.
    """
    levels = np.full(len(times), deal.autocall_barrier, dtype=float)
    if deal.autocall_stepdown_pa:
        elapsed = np.maximum(times - deal.autocall_start_year, 0.0)
        levels = deal.autocall_barrier - deal.autocall_stepdown_pa * elapsed
    return np.where(is_autocall, np.maximum(levels, 0.0), np.nan)


def build_schedule(deal, flat_rate, start_date=None):
    """
    Build an observation grid from deal terms and a flat rate.

    Used by the note builder, where no market discount curve is given.
    The rate is continuously compounded, so DF = exp(-r*T).

    A date carries an autocall test when it falls on the autocall grid at
    or after the first call date. The final date redeems and never calls.
    """
    n = deal.n_observations
    step = 1.0 / deal.coupons_per_year
    times = np.array([(i + 1) * step for i in range(n)])

    call_step = 1.0 / deal.autocalls_per_year
    offsets = (times - deal.autocall_start_year) / call_step
    is_autocall = np.array([
        t >= deal.autocall_start_year - 1e-9
        and abs(o - round(o)) < 1e-6
        and i < n - 1
        for i, (t, o) in enumerate(zip(times, offsets))
    ])

    if start_date is not None:
        labels = [
            (np.datetime64(start_date)
             + np.timedelta64(int(round(t * 365.25)), "D"))
            .astype("datetime64[D]").astype(str)
            for t in times
        ]
    else:
        labels = [f"{t:.2f}y" for t in times]

    return Schedule(
        times=times,
        df_ois=np.exp(-flat_rate * times),
        is_autocall=is_autocall,
        autocall_level=_autocall_levels(deal, times, is_autocall),
        labels=labels,
    )


def schedule_from_table(rows, deal):
    """
    Build a grid from an explicit table of (label, T, DF, is_autocall).

    Used by the worked example, where the discount curve is locked and
    has to be reproduced exactly rather than approximated by a flat rate.
    """
    times = np.array([r[1] for r in rows])
    is_autocall = np.array([r[3] for r in rows])
    return Schedule(
        times=times,
        df_ois=np.array([r[2] for r in rows]),
        is_autocall=is_autocall,
        autocall_level=_autocall_levels(deal, times, is_autocall),
        labels=[r[0] for r in rows],
    )
