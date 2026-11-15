"""
Implied-volatility surfaces, Dupire local volatility, and LSV.

Three things live here.

A **surface**: total implied variance w(k, T) on a grid of log-moneyness
and maturity, with bilinear lookup. Built either from a calibrated
parameter set or from market quotes.

**Dupire local volatility**, read off that surface. Local vol is the
unique one-factor diffusion that reprices every European option on the
surface, and it is what a desk uses when today's smile has to be
matched exactly. Its weakness is the forward smile: it reprices today's
options and then predicts a future smile that flattens far faster than
any market has ever done - which matters here, because an autocallable
is a sequence of forward-starting decisions.

**Local-stochastic volatility**, which is the production answer:
stochastic variance for realistic forward dynamics, multiplied by a
leverage function that pulls the model back onto today's surface. The
leverage is calibrated by the standard conditional-expectation
argument - the model reprices the surface when

    L(S, t)^2 * E[v_t | S_t = S] = sigma_loc(S, t)^2

so L is found by simulating forward, binning paths by spot at each
step, and setting L from the realised conditional average of the
variance. It is a bootstrap: every step depends on the leverage already
fixed at the steps before it.
"""

import numpy as np

import calibrate as cal


class VolSurface:
    """
    Total implied variance on a (log-moneyness, maturity) grid.

    Stores w = sigma^2 * T rather than sigma, because w is what Dupire's
    formula differentiates and it behaves far better near T = 0.
    """

    def __init__(self, ks, ts, w, forwards=None):
        self.ks = np.asarray(ks, dtype=float)
        self.ts = np.asarray(ts, dtype=float)
        self.w = np.asarray(w, dtype=float)       # (n_t, n_k)
        self.forwards = forwards

    @classmethod
    def dense_maturities(cls, t_max, n=26, t_min=0.02):
        """
        Maturity grid for a surface, geometrically spaced from near zero.

        The grid must start well before the first observation date, not
        at it. Dupire's numerator is dw/dT, so a surface whose earliest
        maturity is the first observation has nothing to differentiate
        before that point - the local volatility collapses to its floor
        over the opening months, and everything downstream inherits the
        missing variance. Spacing is geometric because the surface moves
        fastest at the short end.
        """
        return np.geomspace(t_min, t_max, n)

    @classmethod
    def from_params(cls, params, forwards, dfs, maturities,
                    ks=None, n_k=61, k_lo=-1.2, k_hi=0.8):
        """
        Surface implied by a Heston parameter set.

        Using an analytic surface rather than raw quotes keeps it smooth
        and free of the bid-ask jitter that Dupire's second derivative
        would otherwise amplify into noise. A desk smooths market quotes
        for the same reason.
        """
        ks = np.linspace(k_lo, k_hi, n_k) if ks is None else np.asarray(ks)
        w = np.empty((len(maturities), len(ks)))
        for i, (T, F, df) in enumerate(zip(maturities, forwards, dfs)):
            strikes = F * np.exp(ks)
            px = np.atleast_1d(cal.call_price(F, strikes, T, df, params))
            for j, (p, K) in enumerate(zip(px, strikes)):
                iv = cal.implied_vol(p, F, K, T, df)
                w[i, j] = (iv ** 2) * T if np.isfinite(iv) else np.nan
        # fill any wing failures by carrying the nearest good value
        for i in range(w.shape[0]):
            row = w[i]
            good = np.isfinite(row)
            if good.any():
                row[~good] = np.interp(ks[~good], ks[good], row[good])
        return cls(ks, np.asarray(maturities), w, forwards)

    def total_variance(self, k, t):
        """Bilinear lookup of w, flat outside the grid."""
        k = np.clip(np.asarray(k, dtype=float), self.ks[0], self.ks[-1])
        t = float(np.clip(t, self.ts[0], self.ts[-1]))

        i = np.clip(np.searchsorted(self.ts, t) - 1, 0, len(self.ts) - 2)
        t0, t1 = self.ts[i], self.ts[i + 1]
        a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)

        row = (1 - a) * self.w[i] + a * self.w[i + 1]
        return np.interp(k, self.ks, row)

    def implied_vol(self, k, t):
        t = max(float(t), 1e-6)
        return np.sqrt(np.maximum(self.total_variance(k, t), 1e-12) / t)


def dupire_local_vol(surface, k, t, dk=0.02, dt=None):
    """
    Local volatility from total implied variance, Dupire in w-form.

        sigma_loc^2 = dw/dT / [ 1 - (k/w) dw/dk
                        + (1/4)(-1/4 - 1/w + k^2/w^2) (dw/dk)^2
                        + (1/2) d2w/dk2 ]

    Written in total variance and log-moneyness rather than in price
    space: the price form needs second derivatives of option prices in
    strike, which are tiny numbers differenced twice and go to noise
    immediately. This form is the standard one for that reason.

    The denominator is the local-volatility analogue of the butterfly
    no-arbitrage condition; it is floored rather than allowed to go
    negative, which can happen in the far wings of an interpolated
    surface.
    """
    k = np.asarray(k, dtype=float)
    t = max(float(t), 1e-4)

    w = surface.total_variance(k, t)
    w = np.maximum(w, 1e-8)

    # the time bump scales with t: a fixed one is far too coarse at the
    # short end, where the surface moves fastest, and wastefully fine
    # at the long end
    dt = dt if dt is not None else max(0.02 * t, 1e-4)
    t_up = min(t + dt, surface.ts[-1])
    t_dn = max(t - dt, surface.ts[0] * 0.5)
    dwdt = (surface.total_variance(k, t_up)
            - surface.total_variance(k, t_dn)) / max(t_up - t_dn, 1e-8)
    dwdt = np.maximum(dwdt, 1e-8)        # calendar-arbitrage floor

    w_kp = surface.total_variance(k + dk, t)
    w_km = surface.total_variance(k - dk, t)
    dwdk = (w_kp - w_km) / (2 * dk)
    d2wdk2 = (w_kp - 2 * w + w_km) / dk**2

    denom = (1.0
             - k / w * dwdk
             + 0.25 * (-0.25 - 1.0 / w + k**2 / w**2) * dwdk**2
             + 0.5 * d2wdk2)
    denom = np.maximum(denom, 1e-4)

    return np.sqrt(np.maximum(dwdt / denom, 1e-8))


class LocalVolParams:
    """A Dupire local-volatility model built from a surface."""

    label = "Local vol"

    def __init__(self, surface, decrement, n_k=81, n_t=61,
                 k_lo=-1.3, k_hi=0.9):
        self.surface = surface
        self.decrement = decrement
        self.ks = np.linspace(k_lo, k_hi, n_k)
        self.ts = np.geomspace(max(surface.ts[0], 1e-3),
                               surface.ts[-1], n_t)
        self.grid = np.array([dupire_local_vol(surface, self.ks, t)
                              for t in self.ts])
        self.grid = np.clip(self.grid, 0.02, 2.0)

    def describe(self):
        return (f"Dupire local vol, {self.grid.shape[0]}x"
                f"{self.grid.shape[1]} grid")

    def vol(self, k, t):
        """Bilinear lookup, flat outside the grid."""
        k = np.clip(k, self.ks[0], self.ks[-1])
        t = float(np.clip(t, self.ts[0], self.ts[-1]))
        i = np.clip(np.searchsorted(self.ts, t) - 1, 0, len(self.ts) - 2)
        t0, t1 = self.ts[i], self.ts[i + 1]
        a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        row = (1 - a) * self.grid[i] + a * self.grid[i + 1]
        return np.interp(k, self.ks, row)


class LSVParams:
    """
    Local-stochastic volatility: Heston variance times a leverage
    function calibrated so the model reprices the local-vol surface.

    Built by `calibrate_lsv`; holding the leverage grid and the
    stochastic parameters it was calibrated against.
    """

    label = "LSV"

    def __init__(self, heston, local, ks, ts, leverage):
        self.heston = heston
        self.local = local
        self.ks = np.asarray(ks)
        self.ts = np.asarray(ts)
        self.leverage = np.asarray(leverage)        # (n_t, n_k)

    def describe(self):
        return (f"LSV, Heston xi {self.heston.xi:.2f} rho "
                f"{self.heston.rho:+.2f} with calibrated leverage")

    def value(self, k, t):
        k = np.clip(k, self.ks[0], self.ks[-1])
        t = float(np.clip(t, self.ts[0], self.ts[-1]))
        i = np.clip(np.searchsorted(self.ts, t) - 1, 0, len(self.ts) - 2)
        t0, t1 = self.ts[i], self.ts[i + 1]
        a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        row = (1 - a) * self.leverage[i] + a * self.leverage[i + 1]
        return np.interp(k, self.ks, row)


def _conditional_variance(k, v, centres, bandwidth, fallback):
    """
    E[v | k] by Nadaraya-Watson regression.

    Hard bins throw information away: a particle contributes to one bin
    and nothing to its neighbours, so the wings - where particles are
    scarce exactly when the leverage there matters most - are estimated
    from a handful of paths. A Gaussian kernel lets every particle
    contribute to every grid point in proportion to how close it is,
    which is the same estimator with a smooth window instead of a square
    one.

    Where the total kernel weight is still too thin to mean anything,
    the unconditional mean variance is used rather than a neighbour's
    noisy value - a known approximation beats an extrapolated accident.

    The bandwidth must track the particle cloud, not the output grid.
    Early on the paths are packed tightly around the money, and a window
    sized for the grid spans the whole distribution - it would smooth
    the leverage flat exactly where its shape matters. Scaling it to the
    realised dispersion keeps the window proportional to what is
    actually there to average over.
    """
    d = (centres[:, None] - k[None, :]) / bandwidth
    w = np.exp(-0.5 * d * d)

    mass = w.sum(axis=1)
    num = w @ v

    # effective sample size: how many particles a grid point really has
    ess = mass**2 / np.maximum((w**2).sum(axis=1), 1e-300)
    solid = (mass > 1e-8) & (ess >= 30.0)

    out = np.where(mass > 1e-8, num / np.maximum(mass, 1e-300), fallback)
    return np.where(solid, out, fallback)


def calibrate_lsv(schedule, spot, heston, local, decrement,
                  n_paths=100_000, steps_per_year=104, n_grid=61,
                  k_lo=-1.3, k_hi=0.9, seed=11, bandwidth=None,
                  short_refine=4):
    """
    Leverage function by kernel-weighted conditional expectation.

    The model reprices the vanilla surface when

        L(S, t)^2 * E[ v_t | S_t = S ] = sigma_loc(S, t)^2

    which cannot be solved in closed form: the conditional expectation
    depends on the leverage already applied at earlier times. So it is
    bootstrapped forward - at each step the conditional mean variance is
    estimated from the particles, the leverage follows, the paths
    advance using it, and the next step sees the result.

    Two things the simple version gets wrong. Hard bins estimate the
    wings from very few particles; a kernel uses all of them, weighted.
    And a uniform time grid is far too coarse near zero, where both the
    local volatility and the particle distribution change fastest -
    `short_refine` subdivides the opening interval, which is where the
    fit was previously worst.

    Returns an LSVParams.
    """
    import models

    kappa, theta, xi, rho = (heston.kappa, heston.theta,
                             heston.xi, heston.rho)

    grid, _ = models._time_grid(schedule, steps_per_year)
    if short_refine > 1:
        first = grid[grid <= schedule.times[0]]
        refined = np.unique(np.concatenate([
            np.linspace(first[i], first[i + 1], short_refine + 1)
            for i in range(len(first) - 1)]))
        grid = np.unique(np.concatenate([refined,
                                         grid[grid > schedule.times[0]]]))

    dts = np.diff(grid)
    rates = models._step_rates(schedule, grid)
    fwd_log = models._forward_curve(schedule, grid, decrement)

    centres = np.linspace(k_lo, k_hi, n_grid)
    bw_factor = bandwidth or 0.20

    rng = np.random.default_rng(seed)
    half = n_paths // 2
    n = 2 * half

    v = np.full(n, heston.v0)
    x = np.full(n, np.log(spot))

    t0 = float(local.ts[0])
    ts_out = [0.0]
    lev_rows = [np.clip(local.vol(centres, t0)
                        / max(np.sqrt(heston.v0), 1e-8), 0.05, 8.0)]

    for step, dt in enumerate(dts, start=1):
        t = grid[step - 1]
        k = x - (np.log(spot) + fwd_log[step - 1])

        bw = max(bw_factor * float(k.std()), 0.01)
        cond = _conditional_variance(k, v, centres, bw,
                                     fallback=float(v.mean()))
        lev_row = np.clip(local.vol(centres, t)
                          / np.sqrt(np.maximum(cond, 1e-10)), 0.05, 8.0)

        ts_out.append(t)
        lev_rows.append(lev_row)

        lev = np.interp(k, centres, lev_row)

        zv = np.concatenate([(zh := rng.standard_normal(half)), -zh])
        u = np.concatenate([(uh := rng.random(half)), 1.0 - uh])
        v_next = models._qe_variance_step(v, dt, kappa, theta, xi, zv, u)

        v_bar = 0.5 * (v + v_next)
        zi = np.concatenate([(z2 := rng.standard_normal(half)), -z2])

        x = (x + (rates[step - 1] - decrement) * dt
             - 0.5 * lev**2 * v_bar * dt
             + lev * rho / xi * ((v_next - v) - kappa * (theta - v_bar) * dt)
             + lev * np.sqrt(1 - rho**2)
             * np.sqrt(np.maximum(v_bar * dt, 0.0)) * zi)
        v = v_next

    return LSVParams(heston, local, centres, np.asarray(ts_out),
                     np.asarray(lev_rows))
