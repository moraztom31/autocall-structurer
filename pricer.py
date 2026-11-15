"""
Monte Carlo engine for Phoenix autocallable notes.

Per simulated path:
  1. draw the index level at every observation date (GBM, decrement q)
  2. walk the dates in order:
       - pay a coupon where the index is at or above the coupon barrier
       - on an autocall date at or above the autocall barrier: redeem, stop
  3. if never called, apply the capital barrier at final valuation
  4. discount every cash flow with the note's own discount factors

Fair value is the mean of the discounted path values. The gross
structuring margin is the issue price less that fair value.
"""

from dataclasses import dataclass, field

import numpy as np

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
    avg_coupons_paid: float
    pv_coupons: float
    pv_redemption: float

    # per-path arrays, kept for the distribution charts
    exit_times: np.ndarray = field(repr=False, default=None)
    total_payout: np.ndarray = field(repr=False, default=None)
    final_level_pct: np.ndarray = field(repr=False, default=None)

    inputs: dict = field(default_factory=dict)


def simulate_paths(schedule, spot, vol, decrement, n_paths=50_000, seed=42):
    """
    Index levels at every observation date under the risk-neutral measure.

    The GBM is simulated exactly in log space, so there is no
    discretisation error from the time step:

        S(t+dt) = S(t) * exp[ (f - q - vol^2/2)*dt + vol*sqrt(dt)*Z ]

    f is the forward rate over the step and q the index decrement. On a
    decrement index q is a published deduction rather than a dividend
    yield, and it is already reflected in the quoted level - so it is
    modelled here and no separate dividend is subtracted.

    Antithetic sampling: each draw Z is paired with -Z, which halves the
    standard error of the mean at no extra cost.

    Returns an array of shape (n_paths, len(schedule)).
    """
    fwd, dt = schedule.forward_rates()

    rng = np.random.default_rng(seed)
    half = n_paths // 2
    z_half = rng.standard_normal((half, len(schedule)))
    z = np.concatenate([z_half, -z_half], axis=0)

    drift = (fwd - decrement - 0.5 * vol**2) * dt
    diffusion = vol * np.sqrt(dt)

    return spot * np.exp(np.cumsum(drift + diffusion * z, axis=1))


def price(schedule, spot, vol, decrement, credit_spread,
          deal=None, n_paths=50_000, seed=42, paths=None):
    """
    Price a Phoenix note and return value, margin and diagnostics.

    Pass `paths` to reuse an existing simulation - useful when repricing
    the same market scenario against several sets of terms.
    """
    deal = deal or Deal()

    if paths is None:
        paths = simulate_paths(schedule, spot, vol, decrement, n_paths, seed)

    n_obs = len(schedule)
    disc = schedule.df_note(credit_spread)

    lvl_coupon = deal.coupon_barrier * spot
    lvl_autocall = deal.autocall_barrier * spot
    lvl_capital = deal.capital_barrier * spot

    # --- exit date of each path ---------------------------------------
    # A path is called on the FIRST autocall date at or above the barrier.
    call_idx = np.where(schedule.is_autocall)[0]
    if call_idx.size:
        triggered = paths[:, call_idx] >= lvl_autocall
        any_call = triggered.any(axis=1)
        # argmax on booleans returns the first True; it returns 0 when
        # there is none, which the any_call mask then overrides.
        exit_idx = np.where(any_call,
                            call_idx[np.argmax(triggered, axis=1)],
                            n_obs - 1)
    else:
        any_call = np.zeros(paths.shape[0], dtype=bool)
        exit_idx = np.full(paths.shape[0], n_obs - 1)

    # --- coupons ------------------------------------------------------
    # Paid where the date is on or before the exit and the index is at
    # or above the coupon barrier. No memory: a missed coupon is gone.
    alive = np.arange(n_obs)[None, :] <= exit_idx[:, None]
    coupon_paid = alive & (paths >= lvl_coupon)
    pv_coupons = (coupon_paid * deal.coupon_amount * disc[None, :]).sum(axis=1)

    # --- redemption ---------------------------------------------------
    final_level = paths[:, -1]
    redemption = np.where(
        any_call,
        NOMINAL,                                       # called at par
        np.where(final_level >= lvl_capital,
                 NOMINAL,                              # above the barrier
                 NOMINAL * final_level / spot),        # linear loss below
    )
    pv_redemption = redemption * disc[exit_idx]

    pv_total = pv_coupons + pv_redemption
    fair_value = float(pv_total.mean())
    margin = deal.issue_price - fair_value / NOMINAL

    # undiscounted cash received per path, for the outcome distribution
    total_payout = (coupon_paid * deal.coupon_amount).sum(axis=1) + redemption

    return Result(
        fair_value=fair_value,
        margin_pct=margin,
        passes_floor=margin >= deal.margin_floor,
        margin_gap_bp=(margin - deal.margin_floor) * 1e4,
        expected_life=float(schedule.times[exit_idx].mean()),
        prob_autocall=float(any_call.mean()),
        prob_capital_loss=float((~any_call & (final_level < lvl_capital)).mean()),
        avg_coupons_paid=float(coupon_paid.sum(axis=1).mean()),
        pv_coupons=float(pv_coupons.mean()),
        pv_redemption=float(pv_redemption.mean()),
        exit_times=schedule.times[exit_idx],
        total_payout=total_payout,
        final_level_pct=final_level / spot,
        inputs={
            "n_paths": paths.shape[0], "vol": vol, "spot": spot,
            "decrement": decrement, "credit_spread": credit_spread,
            "coupon_rate_pa": deal.coupon_rate_pa,
            "coupon_barrier": deal.coupon_barrier,
            "autocall_barrier": deal.autocall_barrier,
            "capital_barrier": deal.capital_barrier,
            "issue_price": deal.issue_price,
        },
    )


def payoff_at_maturity(deal, levels_pct):
    """
    Redemption at final valuation, for a note that was never called.

    Returns currency per note across a grid of final index levels
    expressed as fractions of the initial level. The function is
    discontinuous at the capital barrier: one tick below it, the holder
    gives up the whole protected amount rather than a sliver of it.
    """
    levels_pct = np.asarray(levels_pct, dtype=float)
    above = levels_pct >= deal.capital_barrier
    redemption = np.where(above, NOMINAL, NOMINAL * levels_pct)
    final_coupon = np.where(levels_pct >= deal.coupon_barrier,
                            deal.coupon_amount, 0.0)
    return redemption + final_coupon
