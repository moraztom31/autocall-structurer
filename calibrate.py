"""
Semi-analytic Heston pricing, and calibration to an implied-vol surface.

Calibrating five parameters by Monte Carlo is not practical: each
objective evaluation would cost a full simulation, and the noise in it
would swamp the gradient. Heston has a known characteristic function,
so vanilla options can be priced by a single numerical integral -
roughly four orders of magnitude faster, and smooth enough for a
least-squares optimiser to work on.

The integral used is Lewis's form, which prices the option in one pass
rather than computing two probabilities separately, and the
characteristic function is written in the "little trap" form of
Albrecher et al. The two formulations are algebraically identical but
the naive one crosses a branch cut of the complex logarithm for longer
maturities, which shows up as the price oscillating wildly in T. This
one does not.
"""

from dataclasses import replace

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

from models import HestonParams

# Gauss-Legendre nodes on (0, U), mapped once and reused.
_GL_N = 160
_GL_U = 200.0
_gl_x, _gl_w = np.polynomial.legendre.leggauss(_GL_N)
_U = 0.5 * _GL_U * (_gl_x + 1.0)
_W = 0.5 * _GL_U * _gl_w


def _char_fn(u, T, params):
    """
    Heston characteristic function of the log-forward, "little trap" form.

    `u` is complex and vectorised; T is a scalar maturity.
    """
    kappa, theta, xi, rho, v0 = (params.kappa, params.theta, params.xi,
                                 params.rho, params.v0)

    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b**2 + (xi**2) * (iu + u**2))

    # g2 = (b - d)/(b + d): the form whose exponential decays, which is
    # what keeps the logarithm on its principal branch as T grows.
    g2 = (b - d) / (b + d)
    e = np.exp(-d * T)

    C = (kappa * theta / xi**2) * ((b - d) * T
                                   - 2.0 * np.log((1.0 - g2 * e)
                                                  / (1.0 - g2)))
    D = ((b - d) / xi**2) * (1.0 - e) / (1.0 - g2 * e)

    return np.exp(C + D * v0)


def call_price(forward, strike, T, df, params):
    """
    Undiscounted-forward European call, by Lewis's integral.

        C = df * [ F - sqrt(F K)/pi * int_0^inf
                   Re( e^{i u k} phi(u - i/2) ) / (u^2 + 1/4) du ]

    with k = ln(F/K). Vectorised over strike.
    """
    K = np.atleast_1d(np.asarray(strike, dtype=float))
    k = np.log(forward / K)

    phi = _char_fn(_U - 0.5j, T, params)              # (n_nodes,)
    integrand = np.real(np.exp(1j * np.outer(k, _U)) * phi[None, :]) \
        / (_U**2 + 0.25)[None, :]
    integral = integrand @ _W

    px = df * (forward - np.sqrt(forward * K) / np.pi * integral)
    return np.maximum(px, 0.0) if K.size > 1 else float(max(px[0], 0.0))


def bs_call(forward, strike, T, df, vol):
    """Black call on a forward."""
    K = np.asarray(strike, dtype=float)
    v = max(float(vol), 1e-9) * np.sqrt(T)
    d1 = np.log(forward / K) / v + 0.5 * v
    return df * (forward * norm.cdf(d1) - K * norm.cdf(d1 - v))


def implied_vol(price_, forward, strike, T, df, lo=1e-4, hi=4.0):
    """Black implied volatility of a call price, by Brent."""
    intrinsic = df * max(forward - strike, 0.0)
    if price_ <= intrinsic + 1e-12:
        return float("nan")
    try:
        return float(brentq(
            lambda v: bs_call(forward, strike, T, df, v) - price_, lo, hi))
    except ValueError:
        return float("nan")


def surface_from_params(forwards, dfs, maturities, moneyness, params):
    """
    Implied-vol grid produced by a parameter set.

    Rows are maturities, columns moneyness (strike / forward). Used to
    build synthetic surfaces for testing, and to evaluate a candidate
    during calibration.
    """
    out = np.empty((len(maturities), len(moneyness)))
    for i, (T, F, df) in enumerate(zip(maturities, forwards, dfs)):
        strikes = np.asarray(moneyness) * F
        px = call_price(F, strikes, T, df, params)
        out[i] = [implied_vol(p, F, K, T, df)
                  for p, K in zip(np.atleast_1d(px), strikes)]
    return out


def calibrate(target_vols, forwards, dfs, maturities, moneyness,
              start=None, weights=None, verbose=False):
    """
    Least-squares fit of Heston to an implied-vol surface.

    Fitting implied volatilities rather than prices matters: prices vary
    by orders of magnitude across strikes, so a price-space objective is
    dominated by the in-the-money options and barely sees the wings -
    which is where the information about rho and xi actually lives, and
    where this product is priced.

    Parameters are optimised in a transformed space (logs for the
    positive ones, arctanh for the correlation) so the optimiser cannot
    step outside the admissible region and need not be told about bounds.

    Returns (params, result) where result carries the RMSE in vol points.
    """
    target = np.asarray(target_vols, dtype=float)
    ok = np.isfinite(target)
    w = np.ones_like(target) if weights is None else np.asarray(weights)

    p0 = start or HestonParams()
    x0 = np.array([np.log(p0.v0), np.log(p0.theta), np.log(p0.kappa),
                   np.log(p0.xi), np.arctanh(np.clip(p0.rho, -0.98, 0.98))])

    def unpack(x):
        return HestonParams(v0=float(np.exp(x[0])),
                            theta=float(np.exp(x[1])),
                            kappa=float(np.exp(x[2])),
                            xi=float(np.exp(x[3])),
                            rho=float(np.tanh(x[4])))

    def resid(x):
        model = surface_from_params(forwards, dfs, maturities,
                                    moneyness, unpack(x))
        r = (model - target) * w
        r[~ok] = 0.0
        return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0).ravel()

    sol = least_squares(resid, x0, method="lm", xtol=1e-10, ftol=1e-10,
                        max_nfev=4000)
    params = unpack(sol.x)
    rmse = float(np.sqrt(np.mean(sol.fun[sol.fun != 0.0] ** 2)))

    if verbose:
        print(f"  calibration RMSE {rmse*100:.3f} vol points, "
              f"{sol.nfev} evaluations")

    return params, {"rmse": rmse, "nfev": sol.nfev, "success": sol.success}
