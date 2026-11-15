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

from dataclasses import dataclass, replace

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


@dataclass(frozen=True)
class BatesParams:
    """
    Heston plus lognormal jumps (Bates).

        dS/S = (r - q - lam*k) dt + sqrt(v) dW1 + (J - 1) dN
        ln J ~ N(mu_j, sigma_j^2),  k = E[J] - 1 = exp(mu_j + sigma_j^2/2) - 1

    The -lam*k term is the compensator: jumps have a non-zero mean
    effect, and without subtracting it the discounted forward stops
    being a martingale and every price is biased.

    Why bother on this product. A diffusion reaches the capital barrier
    by drifting there, which takes time, so short-dated protection looks
    cheap under Heston alone. Jumps let the index arrive at the barrier
    at once - which is how barriers are actually breached - and they put
    a steep skew into short maturities that stochastic volatility cannot
    produce on its own.

    lam     jump intensity, expected jumps per year
    mu_j    mean jump size in log terms; negative for equity crash risk
    sigma_j dispersion of jump sizes
    """

    v0: float = 0.17**2
    theta: float = 0.18**2
    kappa: float = 1.5
    xi: float = 0.45
    rho: float = -0.65
    lam: float = 0.25
    mu_j: float = -0.12
    sigma_j: float = 0.15

    label = "Bates"

    @property
    def compensator(self):
        """k = E[J] - 1."""
        return float(np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1.0)

    def describe(self):
        return (f"Bates, v0 {np.sqrt(self.v0):.1%} · rho {self.rho:+.2f} · "
                f"{self.lam:g} jumps/yr of {self.mu_j:+.0%}")


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

    jumps = isinstance(params, BatesParams)
    lam = params.lam if jumps else 0.0
    mu_j = params.mu_j if jumps else 0.0
    sigma_j = params.sigma_j if jumps else 0.0
    jump_drift = lam * params.compensator if jumps else 0.0

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

        x = (x + (r - decrement - jump_drift) * dt + k0 + k1 * v
             + k2 * v_next
             + np.sqrt(np.maximum(k3 * v + k4 * v_next, 0.0)) * zs)

        if lam > 0.0:
            counts = rng.poisson(lam * dt, size=n)
            hit = counts > 0
            if hit.any():
                tot = counts[hit]
                x[hit] += (mu_j * tot
                           + sigma_j * np.sqrt(tot)
                           * rng.standard_normal(hit.sum()))

        v = v_next

        if k in obs_set:
            out[:, obs_set[k]] = np.exp(x)

    return out


def implied_spot_vol(params):
    """Spot volatility, for labelling and for seeding a Black comparison."""
    if isinstance(params, (HestonParams, BatesParams)):
        return float(np.sqrt(params.v0))
    if isinstance(params, BlackParams):
        return float(params.vol)
    # local vol and LSV carry a surface rather than a single number;
    # their spot volatility is whatever it is at the money, today
    try:
        return float(np.atleast_1d(params.vol(0.0, params.ts[0]))[0])
    except Exception:                                   # noqa: BLE001
        return float(np.sqrt(params.heston.v0))


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
        p = replace(p, v0=p.v0 * scale, theta=p.theta * scale,
                    xi=p.xi * float(np.sqrt(scale)))
    return p


def _forward_curve(schedule, grid, decrement):
    """Deterministic forward factor exp(int (r-q)) at each grid time."""
    rates = _step_rates(schedule, grid)
    dts = np.diff(grid)
    return np.concatenate([[0.0],
                           np.cumsum((rates - decrement) * dts)])


def simulate_local_vol(schedule, spot, params, decrement, n_paths,
                       seed=42, steps_per_year=104):
    """
    Dupire local volatility by log-Euler.

    The volatility is read at each step from the local-vol grid at the
    path's own log-moneyness against the deterministic forward, which is
    the coordinate the surface was built in. Local vol has no exact
    scheme, so the step has to be fine enough: the volatility changes
    with the spot within the step, and a coarse grid systematically
    understates how fast it moves in the wings.
    """
    grid, obs_pos = _time_grid(schedule, steps_per_year)
    dts = np.diff(grid)
    rates = _step_rates(schedule, grid)
    fwd_log = _forward_curve(schedule, grid, decrement)

    rng = np.random.default_rng(seed)
    half = n_paths // 2
    n = 2 * half

    x = np.full(n, np.log(spot))
    out = np.empty((n, len(schedule)))
    obs_set = {int(p): i for i, p in enumerate(obs_pos)}

    for step, dt in enumerate(dts, start=1):
        t = grid[step - 1]
        k = x - (np.log(spot) + fwd_log[step - 1])
        sig = params.vol(k, t)

        z_half = rng.standard_normal(half)
        z = np.concatenate([z_half, -z_half])

        x = (x + (rates[step - 1] - decrement - 0.5 * sig**2) * dt
             + sig * np.sqrt(dt) * z)

        if step in obs_set:
            out[:, obs_set[step]] = np.exp(x)

    return out


def _qe_variance_step(v, dt, kappa, theta, xi, zv, u, psi_c=1.5):
    """
    One Andersen QE step of the Heston variance, given the draws.

    Factored out so Heston and LSV step their variance identically -
    the leverage function changes the spot equation only.
    """
    e = np.exp(-kappa * dt)
    m = theta + (v - theta) * e
    s2 = (v * xi**2 * e * (1 - e) / kappa
          + theta * xi**2 * (1 - e)**2 / (2 * kappa))
    psi = np.where(m > 1e-14, s2 / np.maximum(m**2, 1e-300), np.inf)

    inv_psi = 2.0 / np.maximum(psi, 1e-300)
    b2 = np.maximum(inv_psi - 1 + np.sqrt(np.maximum(
        inv_psi * (inv_psi - 1), 0.0)), 0.0)
    a = m / np.maximum(1 + b2, 1e-300)
    v_lo = a * (np.sqrt(b2) + zv) ** 2

    p = np.clip((psi - 1) / (psi + 1), 0.0, 1.0 - 1e-12)
    beta = np.where(m > 1e-14, (1 - p) / np.maximum(m, 1e-300), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        v_hi = np.where(u <= p, 0.0,
                        np.log(np.maximum((1 - p) / (1 - u), 1e-300))
                        / np.maximum(beta, 1e-300))

    return np.maximum(np.where(psi <= psi_c, v_lo, v_hi), 0.0)


def simulate_lsv(schedule, spot, params, decrement, n_paths, seed=42,
                 steps_per_year=104):
    """
    Local-stochastic volatility: Heston variance scaled by a calibrated
    leverage function L(S, t).

        d ln S = (r - q - L^2 v / 2) dt + L sqrt(v) dW1

    The correlated part of the spot increment is substituted out of the
    variance equation rather than drawn separately:

        sqrt(v) dW2 = [ dv - kappa (theta - v) dt ] / xi

    so the spot step uses the *exact* variance increment the QE scheme
    just produced. Drawing a correlated normal instead would reintroduce
    the discretisation error that QE was chosen to avoid, and would
    understate the skew - which is the entire reason for running this
    model.
    """
    h = params.heston
    kappa, theta, xi, rho = h.kappa, h.theta, h.xi, h.rho

    grid, obs_pos = _time_grid(schedule, steps_per_year)
    dts = np.diff(grid)
    rates = _step_rates(schedule, grid)
    fwd_log = _forward_curve(schedule, grid, decrement)

    rng = np.random.default_rng(seed)
    half = n_paths // 2
    n = 2 * half

    v = np.full(n, h.v0)
    x = np.full(n, np.log(spot))
    out = np.empty((n, len(schedule)))
    obs_set = {int(p): i for i, p in enumerate(obs_pos)}

    for step, dt in enumerate(dts, start=1):
        t = grid[step - 1]

        zv = np.concatenate([(zh := rng.standard_normal(half)), -zh])
        u = np.concatenate([(uh := rng.random(half)), 1.0 - uh])
        v_next = _qe_variance_step(v, dt, kappa, theta, xi, zv, u)

        k = x - (np.log(spot) + fwd_log[step - 1])
        lev = params.value(k, t)

        v_bar = 0.5 * (v + v_next)
        zi = np.concatenate([(zh2 := rng.standard_normal(half)), -zh2])

        x = (x + (rates[step - 1] - decrement) * dt
             - 0.5 * lev**2 * v_bar * dt
             + lev * rho / xi * ((v_next - v) - kappa * (theta - v_bar) * dt)
             + lev * np.sqrt(1 - rho**2)
             * np.sqrt(np.maximum(v_bar * dt, 0.0)) * zi)

        v = v_next
        if step in obs_set:
            out[:, obs_set[step]] = np.exp(x)

    return out


def simulate(schedule, spot, params, decrement, n_paths, seed=42, **kw):
    """Dispatch on the parameter type."""
    name = type(params).__name__
    if name == "LocalVolParams":
        return simulate_local_vol(schedule, spot, params, decrement,
                                  n_paths, seed, **kw)
    if name == "LSVParams":
        return simulate_lsv(schedule, spot, params, decrement,
                            n_paths, seed, **kw)
    if isinstance(params, (HestonParams, BatesParams)):
        return simulate_heston(schedule, spot, params, decrement,
                               n_paths, seed, **kw)
    return simulate_black(schedule, spot, params, decrement, n_paths, seed)


