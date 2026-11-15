"""
Listed option chains: cleaning, forward extraction, surface building.

The hard part of using a real chain is not downloading it. It is that
raw quotes carry stale prints, zero bids, crossed markets and strikes
nobody has traded in weeks, and that the forward - which every implied
volatility depends on - is not given to you.

**The forward comes from put-call parity, not from spot minus
dividends.** For European options

    C(K) - P(K) = DF * (F - K)

so the call-put difference is affine in strike with slope -DF and
intercept DF*F. Regressing it across the liquid strikes recovers the
forward the option market is actually trading, including the dividends
it expects and any borrow. Using spot and a dividend yield off a data
provider instead imports two guesses into every single implied
volatility, and the error lands hardest in the wings - which is where
a barrier product is priced.

Only out-of-the-money quotes are kept. In-the-money options carry the
same information by parity but trade with wider spreads and more stale
prints, so including them adds noise and no signal.
"""

from dataclasses import dataclass, field

import numpy as np

import calibrate as cal
from surface import VolSurface


@dataclass
class ChainSlice:
    """One expiry of a cleaned chain."""

    maturity: float                 # year fraction
    df: float                       # discount factor to that date
    forward: float                  # from put-call parity
    strikes: np.ndarray
    vols: np.ndarray                # Black implied, from mid prices
    n_raw: int = 0
    n_kept: int = 0
    parity_r2: float = float("nan")


@dataclass
class ChainReport:
    """What came back and what was thrown away."""

    slices: list = field(default_factory=list)
    dropped: dict = field(default_factory=dict)
    source: str = "synthetic"
    error: str | None = None

    def summary(self):
        if self.error:
            return self.error
        return (f"{len(self.slices)} expiries, "
                f"{sum(s.n_kept for s in self.slices)} quotes kept of "
                f"{sum(s.n_raw for s in self.slices)}")


def forward_from_parity(strikes, calls, puts, df, weights=None):
    """
    Forward implied by put-call parity, by weighted least squares.

        C - P = DF * (F - K)

    Fitted rather than solved at a single strike: any one pair can be
    stale, and the near-the-money strikes - where the difference is
    smallest relative to the spread - are the least reliable of all.
    Weighting by liquidity and fitting across the chain is what a desk
    does.

    Returns (forward, r2). An r2 well below 1 means the slice is not
    internally consistent and should not be trusted.
    """
    k = np.asarray(strikes, dtype=float)
    d = np.asarray(calls, dtype=float) - np.asarray(puts, dtype=float)
    w = np.ones_like(k) if weights is None else np.asarray(weights,
                                                           dtype=float)
    ok = np.isfinite(d) & np.isfinite(k) & (w > 0)
    if ok.sum() < 3:
        return float("nan"), float("nan")

    k, d, w = k[ok], d[ok], w[ok]
    # d = a + b k, with b = -DF and a = DF*F
    W = np.diag(w)
    X = np.column_stack([np.ones_like(k), k])
    beta = np.linalg.lstsq(W @ X, w * d, rcond=None)[0]
    a, b = beta

    resid = d - (a + b * k)
    ss_res = float((w * resid**2).sum())
    ss_tot = float((w * (d - np.average(d, weights=w))**2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # prefer the slope's own discount factor when it is sane, since it
    # is measured rather than assumed
    df_fit = -b
    df_use = df_fit if 0.5 < df_fit < 1.05 else df
    return float(a / df_use), float(r2)


def clean_slice(maturity, df, strikes, call_bid, call_ask, put_bid,
                put_ask, min_price=0.05, max_spread_frac=0.35,
                k_lo=-1.0, k_hi=0.6):
    """
    One expiry: drop unusable quotes, find the forward, invert to vols.

    A quote is dropped when the bid is zero or missing (nobody is
    buying, so the mid means nothing), when the relative spread is wider
    than `max_spread_frac` (the mid carries less information than its
    own uncertainty), when the price is below `min_price` (a tick or two
    of rounding dominates), or when the strike is outside the moneyness
    band the surface needs.

    Returns (ChainSlice, drop counts).
    """
    K = np.asarray(strikes, dtype=float)
    cb, ca = np.asarray(call_bid, float), np.asarray(call_ask, float)
    pb, pa = np.asarray(put_bid, float), np.asarray(put_ask, float)

    n_raw = K.size
    drops = {}

    cm, pm = 0.5 * (cb + ca), 0.5 * (pb + pa)
    c_ok = np.isfinite(cm) & (cb > 0) & (ca > cb)
    p_ok = np.isfinite(pm) & (pb > 0) & (pa > pb)
    drops["no bid"] = int((~(c_ok | p_ok)).sum())

    with np.errstate(divide="ignore", invalid="ignore"):
        c_spr = np.where(cm > 0, (ca - cb) / cm, np.inf)
        p_spr = np.where(pm > 0, (pa - pb) / pm, np.inf)

    # forward: use strikes where both legs are quoted and tight
    par = c_ok & p_ok & (c_spr < max_spread_frac) & (p_spr < max_spread_frac)
    if par.sum() < 3:
        return None, {"too few parity strikes": int(par.sum())}

    liq = 1.0 / np.maximum(c_spr[par] + p_spr[par], 1e-6)
    fwd, r2 = forward_from_parity(K[par], cm[par], pm[par], df, liq)
    if not np.isfinite(fwd) or fwd <= 0:
        return None, {"forward failed": 1}

    # out-of-the-money side only
    use_call = K >= fwd
    price = np.where(use_call, cm, pm)
    spread = np.where(use_call, c_spr, p_spr)
    ok = np.where(use_call, c_ok, p_ok)

    k = np.log(K / fwd)
    keep = (ok & (spread < max_spread_frac) & (price > min_price)
            & (k > k_lo) & (k < k_hi))
    drops["wide spread"] = int((ok & (spread >= max_spread_frac)).sum())
    drops["too cheap"] = int((ok & (price <= min_price)).sum())
    drops["out of band"] = int((ok & ((k <= k_lo) | (k >= k_hi))).sum())

    if keep.sum() < 4:
        return None, {**drops, "too few after filters": int(keep.sum())}

    vols = []
    for Ki, px, is_c in zip(K[keep], price[keep], use_call[keep]):
        if is_c:
            iv = cal.implied_vol(px, fwd, Ki, maturity, df)
        else:
            # put to call by parity, then invert the call
            iv = cal.implied_vol(px + df * (fwd - Ki), fwd, Ki,
                                 maturity, df)
        vols.append(iv)
    vols = np.asarray(vols)

    good = np.isfinite(vols) & (vols > 0.01) & (vols < 2.5)
    drops["inversion failed"] = int((~good).sum())
    if good.sum() < 4:
        return None, {**drops, "too few vols": int(good.sum())}

    return ChainSlice(maturity=maturity, df=df, forward=fwd,
                      strikes=K[keep][good], vols=vols[good],
                      n_raw=n_raw, n_kept=int(good.sum()),
                      parity_r2=r2), drops


def surface_from_slices(slices, n_k=61, k_lo=-1.0, k_hi=0.6):
    """
    Assemble cleaned slices into a total-variance surface.

    Each slice is interpolated in log-moneyness onto a common grid, in
    total variance rather than volatility, and held flat beyond its own
    last quote. Extrapolating a smile outward is how surfaces acquire
    arbitrage; carrying the edge level is conservative and visible.
    """
    slices = sorted(slices, key=lambda s: s.maturity)
    ks = np.linspace(k_lo, k_hi, n_k)
    w = np.empty((len(slices), n_k))
    for i, s in enumerate(slices):
        k_s = np.log(s.strikes / s.forward)
        order = np.argsort(k_s)
        w_s = (s.vols[order] ** 2) * s.maturity
        w[i] = np.interp(ks, k_s[order], w_s)
    return VolSurface(ks, np.array([s.maturity for s in slices]), w,
                      [s.forward for s in slices])


def calibrate_to_slices(slices, start=None, verbose=False):
    """Fit Heston to cleaned slices, on their own strikes."""
    slices = sorted(slices, key=lambda s: s.maturity)
    mny = np.linspace(0.65, 1.25, 13)

    target = np.empty((len(slices), mny.size))
    for i, s in enumerate(slices):
        m_s = s.strikes / s.forward
        order = np.argsort(m_s)
        target[i] = np.interp(mny, m_s[order], s.vols[order],
                              left=np.nan, right=np.nan)

    return cal.calibrate(target,
                         [s.forward for s in slices],
                         [s.df for s in slices],
                         [s.maturity for s in slices],
                         mny, start=start, verbose=verbose)


def fetch_chain(ticker, rate=0.03, max_expiries=8, min_days=20,
                max_days=800):
    """
    Pull and clean a listed option chain.

    Never raises: any failure comes back as a ChainReport carrying an
    error, so the interface can fall back rather than break. Index
    options are often unavailable on free feeds even when the index
    itself quotes; a liquid ETF on the same underlying is the usual
    substitute and the caller is told which was used.
    """
    try:
        import yfinance as yf
    except ImportError:
        return ChainReport(error="yfinance is not installed.")

    try:
        tk = yf.Ticker(ticker)
        expiries = list(tk.options or [])
        if not expiries:
            return ChainReport(error=f"No listed options for {ticker}.")

        import datetime as _dt
        today = _dt.date.today()
        slices, dropped = [], {}

        for e in expiries:
            d = (_dt.date.fromisoformat(e) - today).days
            if d < min_days or d > max_days:
                continue
            T = d / 365.25
            df = float(np.exp(-rate * T))

            ch = tk.option_chain(e)
            c = ch.calls.set_index("strike")
            p = ch.puts.set_index("strike")
            K = np.array(sorted(set(c.index) & set(p.index)), dtype=float)
            if K.size < 6:
                continue

            sl, dr = clean_slice(
                T, df, K,
                c.loc[K, "bid"].to_numpy(float),
                c.loc[K, "ask"].to_numpy(float),
                p.loc[K, "bid"].to_numpy(float),
                p.loc[K, "ask"].to_numpy(float))
            for k_, v_ in dr.items():
                dropped[k_] = dropped.get(k_, 0) + v_
            if sl is not None:
                slices.append(sl)
            if len(slices) >= max_expiries:
                break

        if not slices:
            return ChainReport(dropped=dropped,
                               error=f"No usable expiries for {ticker}.")
        return ChainReport(slices=slices, dropped=dropped, source="live")
    except Exception as exc:                            # noqa: BLE001
        return ChainReport(error=f"Could not fetch {ticker}: {exc}")


def synthetic_chain(params, forward, maturities, rate=0.03,
                    n_strikes=25, spread_frac=0.04, junk_frac=0.15,
                    seed=5):
    """
    A chain with the defects of a real one, for testing the pipeline.

    Prices come from the characteristic function, then acquire a
    bid-ask, tick rounding, a sprinkling of zero bids, and a few stale
    crossed quotes - the things that break naive processing. The true
    parameters and forward are known, so the whole path from quotes to
    calibrated model can be checked rather than assumed.
    """
    rng = np.random.default_rng(seed)
    out = []
    for T in maturities:
        df = float(np.exp(-rate * T))
        F = forward * np.exp(0.0)
        K = np.linspace(0.55 * F, 1.45 * F, n_strikes)

        c = np.atleast_1d(cal.call_price(F, K, T, df, params))
        p = c - df * (F - K)

        def quote(mid):
            half = 0.5 * spread_frac * np.maximum(mid, 1e-6)
            bid = np.round(np.maximum(mid - half, 0.0), 2)
            ask = np.round(mid + half, 2)
            junk = rng.random(mid.shape) < junk_frac
            bid = np.where(junk, 0.0, bid)             # no bid
            crossed = rng.random(mid.shape) < 0.03
            ask = np.where(crossed, bid * 0.5, ask)    # stale crossed
            return bid, ask

        cb, ca = quote(c)
        pb, pa = quote(p)
        out.append((T, df, K, cb, ca, pb, pa, F))
    return out
