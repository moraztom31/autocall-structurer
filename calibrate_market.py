"""
Calibrate Heston to a live listed option chain.

    python calibrate_market.py SPY
    python calibrate_market.py ^STOXX50E --rate 0.025

Prints the cleaned chain, the forwards implied by put-call parity, the
fitted parameters and the quality of the fit. With no network, or no
usable quotes, it says so and prices nothing.

A note on tickers: free feeds carry options for large ETFs far more
reliably than for the indices themselves. SPY, QQQ, IWM and EZU
usually work; ^STOXX50E and ^GSPC often return nothing. An ETF on the
same underlying is a reasonable proxy for the shape of the surface,
which is what matters here - but it is a proxy, and its dividend and
borrow differ from the index's.
"""

import argparse

import numpy as np

from calibrate import surface_from_params
from models import HestonParams
from option_chain import calibrate_to_slices, fetch_chain


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ticker", nargs="?", default="SPY")
    ap.add_argument("--rate", type=float, default=0.03,
                    help="flat continuously compounded discount rate")
    ap.add_argument("--max-expiries", type=int, default=8)
    args = ap.parse_args()

    print(f"\nFetching the option chain for {args.ticker}…")
    rep = fetch_chain(args.ticker, rate=args.rate,
                      max_expiries=args.max_expiries)

    if rep.error:
        print(f"\n  {rep.error}")
        print("  Nothing was calibrated. Try a liquid ETF such as SPY, "
              "or check the connection.")
        return

    print(f"  {rep.summary()}")
    if rep.dropped:
        drops = ", ".join(f"{v} {k}" for k, v in sorted(rep.dropped.items())
                          if v)
        if drops:
            print(f"  dropped: {drops}")

    print(f"\n{'Expiry':>8}{'Forward':>12}{'Parity R2':>12}"
          f"{'Quotes':>9}{'ATM vol':>10}")
    for s in rep.slices:
        k = np.log(s.strikes / s.forward)
        atm = float(np.interp(0.0, k[np.argsort(k)],
                              s.vols[np.argsort(k)]))
        print(f"{s.maturity:>7.2f}y{s.forward:>12,.2f}"
              f"{s.parity_r2:>12.5f}{s.n_kept:>9}{atm:>9.2%}")

    print("\n  Forwards are implied by put-call parity, not assumed from "
          "spot\n  less a dividend estimate. An R2 well below 1 would "
          "mean the slice\n  is not internally consistent.")

    print("\nCalibrating Heston…")
    fit, info = calibrate_to_slices(
        rep.slices,
        start=HestonParams(v0=0.04, theta=0.04, kappa=1.5, xi=0.5,
                           rho=-0.6),
        verbose=True)

    print(f"\n  v0     {np.sqrt(fit.v0):>8.2%}  (spot vol)")
    print(f"  theta  {np.sqrt(fit.theta):>8.2%}  (long-run vol)")
    print(f"  kappa  {fit.kappa:>8.3f}")
    print(f"  xi     {fit.xi:>8.3f}  (vol of vol)")
    print(f"  rho    {fit.rho:>+8.3f}  (skew)")
    print(f"  Feller 2*kappa*theta - xi^2 = {fit.feller:+.4f}"
          f"{'  (satisfied)' if fit.feller > 0 else '  (violated: the '
                                                    'variance can reach zero)'}")

    if info["rmse"] > 0.01:
        print(f"\n  The fit is loose at {info['rmse']*100:.2f} vol points. "
              "Heston has one\n  skew shape per maturity and real "
              "surfaces often need more;\n  treat the parameters as "
              "indicative.")

    print("\n  rho is the parameter an autocallable price depends on, and "
          "it is\n  the best-determined of the five. kappa is the least - "
          "it moves by\n  a factor of two under ordinary bid-ask noise.")


if __name__ == "__main__":
    main()
