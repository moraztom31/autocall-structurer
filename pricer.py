"""
Monte Carlo engine for autocallable notes.

Per simulated path:
  1. draw the index level at every observation date (GBM, decrement q)
  2. walk the dates in order:
       - coupons, under the note's own coupon mechanics
       - on an autocall date at or above that date's trigger: redeem,
         and nothing is paid afterwards
  3. if never called, apply the capital barrier at final valuation
  4. discount every cash flow with the note's own discount factors

Fair value is the mean of the discounted path values. The gross
structuring margin is the issue price less that fair value.
"""

from dataclasses import dataclass, field

import numpy as np

import models
from deal import NOMINAL, Deal


@dataclass
class Result:
    """Pricing output plus the diagnostics the interface charts."""

    fair_value: float            # currency per note
    margin_pct: float            # gross structuring margin, % of nominal
    passes_floor: bool
    margin_gap_bp: float         # distance to the floor, basis points

    expected_life: float         # mean time to exit, years
    prob_autocall: float
    prob_capital_loss: float
    avg_coupons_paid: float      # in coupon units, not payment dates
    pv_coupons: float
    pv_redemption: float

    # per-path arrays, for the distribution charts
    exit_times: np.ndarray = field(repr=False, default=None)
    total_payout: np.ndarray = field(repr=False, default=None)
    final_level_pct: np.ndarray = field(repr=False, default=None)

    inputs: dict = field(default_factory=dict)


def simulate_paths(schedule, spot, model, decrement, n_paths=50_000,
                   seed=42, **kw):
    """
    Index levels at every observation date under the risk-neutral measure.

    `model` is a BlackParams or a HestonParams; the dynamics live in
    models.py. On a decrement index the deduction q is a published rule
    rather than a dividend yield, and it is already reflected in the
    quoted level - so it enters the drift and no separate dividend is
    subtracted.

    Returns an array of shape (n_paths, len(schedule)).
    """
    return models.simulate(schedule, spot, model, decrement, n_paths,
                           seed, **kw)


def _exit_index(paths, schedule, spot):
    """
    First autocall date breached on each path, else final valuation.

    The trigger is read per date, so a step-down schedule is handled
    without any special case: each date simply has its own level.
    """
    n_obs = paths.shape[1]
    call_idx = np.where(schedule.is_autocall)[0]

    if call_idx.size == 0:
        return (np.zeros(paths.shape[0], dtype=bool),
                np.full(paths.shape[0], n_obs - 1))

    levels = schedule.autocall_level[call_idx] * spot
    triggered = paths[:, call_idx] >= levels[None, :]
    any_call = triggered.any(axis=1)
    # argmax on booleans gives the first True; it returns 0 when there
    # is none, which the any_call mask then overrides.
    exit_idx = np.where(any_call,
                        call_idx[np.argmax(triggered, axis=1)],
                        n_obs - 1)
    return any_call, exit_idx


def _coupon_units(paths, schedule, spot, deal, exit_idx, any_call):
    """
    Coupon units falling due at each observation date, per path.

    Three mechanics, all resolved without a Python loop over dates.

    Plain:   one unit on every date the barrier is met; a missed coupon
             is gone for good.
    Memory:  on a date the barrier is met, every coupon missed since the
             last payment is recovered too. Because meeting the barrier
             always catches the holder up to the full count, the running
             total after date i is i+1 if the barrier was met at or
             before i - a running maximum - and each date's payment is
             the increment in that total.
    Athena:  nothing falls due until the note redeems; at the exit date
             every elapsed period is paid at once, provided the note was
             called or finished at or above the barrier.
    """
    n_paths, n_obs = paths.shape
    idx = np.arange(n_obs)
    alive = idx[None, :] <= exit_idx[:, None]
    met = paths >= deal.effective_coupon_barrier * spot

    if deal.is_athena:
        units = np.zeros((n_paths, n_obs))
        final_ok = met[np.arange(n_paths), exit_idx]
        pay = any_call | final_ok
        units[np.arange(n_paths), exit_idx] = np.where(pay, exit_idx + 1, 0)
        return units

    if deal.coupon_memory:
        reached = np.maximum.accumulate((idx[None, :] + 1) * met, axis=1)
        units = np.diff(np.hstack([np.zeros((n_paths, 1)), reached]), axis=1)
        return units * alive

    return (met & alive).astype(float)


def _redemption(paths, spot, deal, exit_idx, any_call):
    """
    Principal repaid on each path.

    Called notes redeem at par. At maturity, a note at or above the
    capital barrier redeems at par; below it the holder takes the index
    loss. An airbag measures that loss from a lower strike instead of
    from the initial level, which softens the first part of the fall -
    though it does not remove the drop at the barrier itself.
    """
    final_level = paths[:, -1]
    at_maturity = final_level >= deal.capital_barrier * spot

    if deal.airbag:
        strike = max(deal.airbag_strike, 1e-9) * spot
        below = np.minimum(NOMINAL * final_level / strike, NOMINAL)
    else:
        below = NOMINAL * final_level / spot

    return np.where(any_call, NOMINAL,
                    np.where(at_maturity, NOMINAL, below))


def price(schedule, spot, model, decrement, credit_spread,
          deal=None, n_paths=50_000, seed=42, paths=None, **sim_kw):
    """
    Price an autocallable note and return value, margin and diagnostics.

    `model` is a BlackParams or a HestonParams. Pass `paths` to reuse an
    existing simulation - useful when repricing the same market scenario
    against several sets of terms.
    """
    deal = deal or Deal()

    if paths is None:
        paths = simulate_paths(schedule, spot, model, decrement,
                               n_paths, seed, **sim_kw)

    disc = schedule.df_note(credit_spread)

    any_call, exit_idx = _exit_index(paths, schedule, spot)
    units = _coupon_units(paths, schedule, spot, deal, exit_idx, any_call)

    coupon_cash = units * deal.coupon_amount
    pv_coupons = (coupon_cash * disc[None, :]).sum(axis=1)

    redemption = _redemption(paths, spot, deal, exit_idx, any_call)
    pv_redemption = redemption * disc[exit_idx]

    fair_value = float((pv_coupons + pv_redemption).mean())
    margin = deal.issue_price - fair_value / NOMINAL

    final_level = paths[:, -1]
    loss = ~any_call & (final_level < deal.capital_barrier * spot)

    return Result(
        fair_value=fair_value,
        margin_pct=margin,
        passes_floor=margin >= deal.margin_floor,
        margin_gap_bp=(margin - deal.margin_floor) * 1e4,
        expected_life=float(schedule.times[exit_idx].mean()),
        prob_autocall=float(any_call.mean()),
        prob_capital_loss=float(loss.mean()),
        avg_coupons_paid=float(units.sum(axis=1).mean()),
        pv_coupons=float(pv_coupons.mean()),
        pv_redemption=float(pv_redemption.mean()),
        exit_times=schedule.times[exit_idx],
        total_payout=coupon_cash.sum(axis=1) + redemption,
        final_level_pct=final_level / spot,
        inputs={
            "n_paths": paths.shape[0], "spot": spot,
            "model": getattr(model, "label", "?"),
            "model_detail": model.describe(),
            "spot_vol": models.implied_spot_vol(model),
            "decrement": decrement, "credit_spread": credit_spread,
            "style": deal.style, "coupon_rate_pa": deal.coupon_rate_pa,
            "coupon_barrier": deal.effective_coupon_barrier,
            "coupon_memory": deal.coupon_memory,
            "autocall_barrier": deal.autocall_barrier,
            "stepdown": deal.autocall_stepdown_pa,
            "capital_barrier": deal.capital_barrier,
            "airbag": deal.airbag, "issue_price": deal.issue_price,
        },
    )


def payoff_at_maturity(deal, levels_pct, n_obs=None):
    """
    Redemption at final valuation, for a note that was never called.

    Returns currency per note across a grid of final index levels, as
    fractions of the initial level. The function is discontinuous at the
    capital barrier: one tick below it the holder gives up the whole
    protected amount rather than a sliver of it. An airbag lifts the
    segment below the barrier, and closes the gap altogether when its
    strike sits at or below the barrier.

    The coupon component is what falls due on the final date: one coupon
    for a Phoenix, the whole accrued stream for an Athena.
    """
    levels_pct = np.asarray(levels_pct, dtype=float)
    n_obs = n_obs or deal.n_observations

    above = levels_pct >= deal.capital_barrier
    if deal.airbag:
        strike = max(deal.airbag_strike, 1e-9)
        below = np.minimum(NOMINAL * levels_pct / strike, NOMINAL)
    else:
        below = NOMINAL * levels_pct

    redemption = np.where(above, NOMINAL, below)

    per_coupon = deal.coupon_amount * (n_obs if deal.is_athena else 1)
    coupon = np.where(levels_pct >= deal.effective_coupon_barrier,
                      per_coupon, 0.0)

    return redemption + coupon
