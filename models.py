"""
Index dynamics: flat Black, and Heston stochastic volatility.

Why two. A flat Black volatility gives every strike the same vol, so
the model has no skew at all. An autocallable holder is short a
down-and-in put struck at the capital barrier, and the price of that
put is mostly a statement about the left tail - exactly where a flat
surface is wrong. Pricing the same note under both dynamics is the
point: the gap between them is the model risk the flat number hides.

What this is not. Heston here is parameterised, not calibrated: there
is no option surface in this project to fit, and a decrement index has
almost no listed options to fit to anyway - desks proxy the surface
from the parent index. So the Heston price is not "the right answer"
where Black was "the wrong one". It is a second opinion with a skew in
it, and the honest output is the spread between the two plus how that
spread moves with the skew parameters.

Production autocall books go further again, to local-stochastic
volatility: local vol alone reprices today's smile but misstates
forward skew, and stochastic vol alone does not reprice the smile
exactly. Heston is the step that makes the question visible, not the
end of it.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BlackParams:
    """Flat Black volatility: one number for every strike and maturity."""

    vol: float = 0.17

    label = "Black, flat"

    def describe(self):
        return f"flat Black at {self.vol:.1%}"


@dataclass(frozen=True)
class HestonParams:
    """
    Heston stochastic volatility.

        dv = kappa (theta - v) dt + xi sqrt(v) dW2
        corr(dW1, dW2) = rho

    v0     initial variance; sqrt(v0) is the spot volatility
    theta  long-run variance the process reverts to
    kappa  speed of that reversion
    xi     volatility of volatility - controls smile curvature
    rho    spot/vol correlation - controls skew; negative for equities,
           because volatility rises as the index falls
    """

    v0: float = 0.17**2
    theta: float = 0.18**2
    kappa: float = 1.5
    xi: float = 0.55
    rho: float = -0.70

    label = "Heston"

    @property
    def feller(self):
        """2*kappa*theta - xi^2. Below zero the variance can reach zero."""
        return 2 * self.kappa * self.theta - self.xi**2

    def describe(self):
        return (f"Heston, v0 {np.sqrt(self.v0):.1%} · theta "
                f"{np.sqrt(self.theta):.1%} · kappa {self.kappa:g} · "
                f"xi {self.xi:g} · rho {self.rho:+.2f}")


def _time_grid(schedule, steps_per_year):
    """
    A simulation grid fine enough for the variance process, carrying the
    index of each observation date within it.

    Black needs no intermediate steps - the GBM is exact over any
    interval - but Heston's variance has to be stepped, and the barriers
    are still only read on observation dates.
    """
    times = schedule.times
    grid = [0.0]
    obs_pos = []
    prev = 0.0
    for t in times:
        n = max(1, int(np.ceil((t - prev) * steps_per_year)))
        grid.extend(np.linspace(prev, t, n + 1)[1:])
        obs_pos.append(len(grid) - 1)
        prev = t
    return np.asarray(grid), np.asarray(obs_pos)


def _step_rates(schedule, grid):
    """Piecewise-constant risk-free forward over each simulation step."""
    fwd, _ = schedule.forward_rates()
    edges = np.concatenate([[0.0], schedule.times])
    mid = 0.5 * (grid[1:] + grid[:-1])
    idx = np.clip(np.searchsorted(edges, mid, side="right") - 1,
                  0, len(fwd) - 1)
    return fwd[idx]


def simulate_black(schedule, spot, params, decrement, n_paths, seed=42):
    """
    Exact GBM sampling at the observation dates.

        S(t+dt) = S(t) exp[(f - q - vol^2/2) dt + vol sqrt(dt) Z]

    Antithetic sampling pairs each draw with its negative, halving the
    standard error of the mean at no extra cost.
    """
    fwd, dt = schedule.forward_rates()
    vol = params.vol

    rng = np.random.default_rng(seed)
    half = n_paths // 2
    z_half = rng.standard_normal((half, len(schedule)))
    z = np.concatenate([z_half, -z_half], axis=0)

    drift = (fwd - decrement - 0.5 * vol**2) * dt
    return spot * np.exp(np.cumsum(drift + vol * np.sqrt(dt) * z, axis=1))


def simulate_heston(schedule, spot, params, decrement, n_paths, seed=42,
                    steps_per_year=52):
    """
    Heston paths by Andersen's QE scheme, sampled at observation dates.

    The variance is stepped with the quadratic-exponential scheme rather
    than Euler: a Euler step can push the variance negative, and the
    usual fixes (truncate, reflect) bias the very left tail this product
    is priced on. QE instead matches the first two moments of the exact
    transition, switching between a squared-normal and an
    exponential-with-mass-at-zero form depending on how peaked the
    conditional distribution is.

    The log-spot uses the matching Andersen discretisation, which
    substitutes the exact variance increment into the integrated
    variance rather than approximating it - that is what keeps the
    correlation, and so the skew, from being understated.

    Returns an array of shape (n_paths, len(schedule)).
    """
    grid, obs_pos = _time_grid(schedule, steps_per_year)
    dts = np.diff(grid)
    rates = _step_rates(schedule, grid)

    kappa, theta, xi, rho = (params.kappa, params.theta,
                             params.xi, params.rho)

    rng = np.random.default_rng(seed)
    half = n_paths // 2
    n = 2 * half

    v = np.full(n, params.v0)
    x = np.full(n, np.log(spot))          # log spot
    out = np.empty((n, len(schedule)))

    gamma1 = gamma2 = 0.5                  # central discretisation
    psi_c = 1.5                            # QE switching threshold
    obs_set = {int(p): i for i, p in enumerate(obs_pos)}

    for k, dt in enumerate(dts, start=1):
        e = np.exp(-kappa * dt)

        # --- variance: QE ---------------------------------------------
        m = theta + (v - theta) * e
        s2 = (v * xi**2 * e * (1 - e) / kappa
              + theta * xi**2 * (1 - e)**2 / (2 * kappa))
        psi = np.where(m > 1e-14, s2 / np.maximum(m**2, 1e-300), np.inf)

        zv_half = rng.standard_normal((half,))
        zv = np.concatenate([zv_half, -zv_half])
        u_half = rng.random((half,))
        u = np.concatenate([u_half, 1.0 - u_half])

        # peaked branch: v' = a (b + Z)^2
        inv_psi = 2.0 / np.maximum(psi, 1e-300)
        b2 = np.maximum(inv_psi - 1 + np.sqrt(np.maximum(
            inv_psi * (inv_psi - 1), 0.0)), 0.0)
        a = m / np.maximum(1 + b2, 1e-300)
        v_lo = a * (np.sqrt(b2) + zv) ** 2

        # flat branch: an atom at zero plus an exponential tail
        p = np.clip((psi - 1) / (psi + 1), 0.0, 1.0 - 1e-12)
        beta = np.where(m > 1e-14, (1 - p) / np.maximum(m, 1e-300), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            v_hi = np.where(u <= p, 0.0,
                            np.log(np.maximum((1 - p) / (1 - u), 1e-300))
                            / np.maximum(beta, 1e-300))

        v_next = np.where(psi <= psi_c, v_lo, v_hi)
        v_next = np.maximum(v_next, 0.0)

        # --- log spot: Andersen, with the exact variance increment ----
        r = rates[k - 1]
        k0 = -rho * kappa * theta * dt / xi
        k1 = gamma1 * dt * (kappa * rho / xi - 0.5) - rho / xi
        k2 = gamma2 * dt * (kappa * rho / xi - 0.5) + rho / xi
        k3 = gamma1 * dt * (1 - rho**2)
        k4 = gamma2 * dt * (1 - rho**2)

        zs_half = rng.standard_normal((half,))
        zs = np.concatenate([zs_half, -zs_half])

        x = (x + (r - decrement) * dt + k0 + k1 * v + k2 * v_next
             + np.sqrt(np.maximum(k3 * v + k4 * v_next, 0.0)) * zs)
        v = v_next

        if k in obs_set:
            out[:, obs_set[k]] = np.exp(x)

    return out


def simulate(schedule, spot, params, decrement, n_paths, seed=42, **kw):
    """Dispatch on the parameter type."""
    if isinstance(params, HestonParams):
        return simulate_heston(schedule, spot, params, decrement,
                               n_paths, seed, **kw)
    return simulate_black(schedule, spot, params, decrement, n_paths, seed)


def implied_spot_vol(params):
    """Spot volatility, for labelling and for seeding a Black comparison."""
    return (float(np.sqrt(params.v0)) if isinstance(params, HestonParams)
            else float(params.vol))


def atm_implied_vol(schedule, spot, params, decrement, maturity_idx=-1,
                    n_paths=120_000, seed=7):
    """
    Black implied volatility of an at-the-money-forward option under
    `params`, at one observation date.

    Priced off the simulated terminal distribution and inverted with
    Brent, which needs no closed form and so works for any dynamics the
    simulator supports.
    """
    from scipy.optimize import brentq
    from scipy.stats import norm

    i = maturity_idx % len(schedule)
    T, df = schedule.times[i], schedule.df_ois[i]
    fwd = spot / df * np.exp(-decrement * T)

    s = simulate(schedule, spot, params, decrement, n_paths, seed)[:, i]
    px = df * np.maximum(fwd - s, 0.0).mean()      # ATM-forward put

    def bs_put(v):
        d1 = (np.log(fwd / fwd) + 0.5 * v * v * T) / (v * np.sqrt(T))
        d2 = d1 - v * np.sqrt(T)
        return df * fwd * (norm.cdf(-d2) - norm.cdf(-d1))

    try:
        return float(brentq(lambda v: bs_put(v) - px, 1e-3, 2.0))
    except ValueError:
        return float("nan")


def match_atm_vol(schedule, spot, params, decrement, target_vol,
                  maturity_idx=-1, n_paths=120_000, iters=4):
    """
    Scale a Heston parameter set so its ATM implied volatility matches a
    target, leaving the shape of the smile alone.

    Comparing a flat Black price against an arbitrary Heston price
    conflates two different things: the overall level of volatility, and
    the skew. Only the second is the question here. Scaling v0 and theta
    together by a common factor moves the level while leaving rho and
    the ratio xi/sqrt(theta) - the skew and the curvature - essentially
    where they were, so the remaining gap between the two prices is the
    skew effect and nothing else.

    Returns a new HestonParams.
    """
    p = params
    for _ in range(iters):
        got = atm_implied_vol(schedule, spot, p, decrement, maturity_idx,
                              n_paths)
        if not np.isfinite(got) or got <= 0:
            break
        scale = (target_vol / got) ** 2
        if abs(scale - 1.0) < 1e-3:
            break
        p = HestonParams(v0=p.v0 * scale, theta=p.theta * scale,
                         kappa=p.kappa, xi=p.xi * np.sqrt(scale),
                         rho=p.rho)
    return p
