"""
Risk measures for an autocallable note.

Every Greek here is a bumped finite difference computed under **common
random numbers**: the same seed, and so the same draws, are used for
the base and the bumped run. Without that, the difference between two
independent Monte Carlo prices is mostly sampling noise - a 1 % spot
bump moves the price by a few tenths of a currency unit, while the
standard error of either price alone is larger than that. Sharing the
draws makes the noise common to both legs, where it cancels.

The spot bump is applied by rescaling the simulated paths rather than
re-simulating. For Black that is exact, since the GBM is a
multiplicative function of its draws. For the stochastic-volatility
models it treats the variance path as unchanged under a spot bump,
which is the sticky-strike convention and the usual desk assumption for
a delta on these books.
"""

from dataclasses import dataclass, replace

import numpy as np

from models import BlackParams, HestonParams
from pricer import price, simulate_paths


@dataclass
class Greeks:
    """Per note, in currency units unless stated."""

    delta: float             # change in value per 1 % move in spot
    gamma: float             # change in delta per 1 % move in spot
    vega: float              # change in value per 1 volatility point
    rho_rate: float          # change in value per 1 bp of credit spread
    theta_days: float        # change in value per calendar day of decay

    spot_ladder: list        # (spot shift, value, delta) across a range
    vega_ladder: list        # (spot shift, vega) across the same range


def _bumped_vol(model, bump):
    """Shift the whole volatility level of either model by `bump`."""
    if isinstance(model, BlackParams):
        return BlackParams(model.vol + bump)
    base = np.sqrt(model.v0)
    k = ((base + bump) / base) ** 2
    return replace(model, v0=model.v0 * k, theta=model.theta * k,
                   xi=model.xi * float(np.sqrt(k)))


def _value(schedule, spot, model, decrement, spread, deal, paths):
    return price(schedule, spot, model, decrement, spread, deal,
                 paths=paths).fair_value


def compute(schedule, spot, model, decrement, spread, deal,
            n_paths=50_000, seed=42, spot_bump=0.01, vol_bump=0.01,
            ladder=(-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20)):
    """
    Delta, gamma, vega, credit rho and theta, plus the ladders a desk
    would actually look at.

    Spot sensitivities are quoted per 1 % move rather than per index
    point, which is how a structured-note book is usually risk-managed
    and keeps the number readable across underlyings.
    """
    paths = simulate_paths(schedule, spot, model, decrement, n_paths, seed)

    def val_at(scale, p=None, m=None, s=None):
        return _value(schedule, spot, m or model, decrement,
                      spread if s is None else s, deal,
                      (p if p is not None else paths) * scale)

    base = val_at(1.0)
    up = val_at(1.0 + spot_bump)
    down = val_at(1.0 - spot_bump)

    delta = (up - down) / 2.0
    gamma = up - 2.0 * base + down

    # vega: re-simulate under a bumped vol, same seed
    v_up = _bumped_vol(model, vol_bump)
    v_dn = _bumped_vol(model, -vol_bump)
    p_up = simulate_paths(schedule, spot, v_up, decrement, n_paths, seed)
    p_dn = simulate_paths(schedule, spot, v_dn, decrement, n_paths, seed)
    vega = (_value(schedule, spot, v_up, decrement, spread, deal, p_up)
            - _value(schedule, spot, v_dn, decrement, spread, deal, p_dn)) / 2.0

    # credit rho: 1 bp on the issuer spread
    rho_rate = (val_at(1.0, s=spread + 1e-4)
                - val_at(1.0, s=spread - 1e-4)) / 2.0

    # theta: one calendar day of carry, approximated by rolling the
    # discount curve forward and leaving the paths where they are
    dt = 1.0 / 365.25
    sched_fwd = replace(schedule,
                        df_ois=schedule.df_ois * np.exp(
                            spread * 0 + _short_rate(schedule) * dt))
    theta_days = (_value(sched_fwd, spot, model, decrement, spread, deal,
                         paths) - base)

    spot_ladder, vega_ladder = [], []
    for sh in ladder:
        s_scale = 1.0 + sh
        v0 = val_at(s_scale)
        d = (val_at(s_scale * (1 + spot_bump))
             - val_at(s_scale * (1 - spot_bump))) / 2.0
        pu = simulate_paths(schedule, spot, v_up, decrement, n_paths, seed)
        pd = simulate_paths(schedule, spot, v_dn, decrement, n_paths, seed)
        vg = (_value(schedule, spot, v_up, decrement, spread, deal,
                     pu * s_scale)
              - _value(schedule, spot, v_dn, decrement, spread, deal,
                       pd * s_scale)) / 2.0
        spot_ladder.append((sh, v0, d))
        vega_ladder.append((sh, vg))

    return Greeks(delta=delta, gamma=gamma, vega=vega, rho_rate=rho_rate,
                  theta_days=theta_days, spot_ladder=spot_ladder,
                  vega_ladder=vega_ladder)


def _short_rate(schedule):
    """Front-end forward rate, used for the one-day carry roll."""
    fwd, _ = schedule.forward_rates()
    return float(fwd[0])
