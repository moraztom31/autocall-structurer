"""
Command-line analysis of the worked example.

    python main.py
"""

import numpy as np

from deal import Deal
from pricer import price
import worked_example as wx

RULE = "=" * 70


def section(title):
    print(f"\n{title}\n{RULE}")


def run(deal=None, vol=None, spread=None):
    return price(wx.SCHEDULE, wx.SPOT, vol or wx.VOL, wx.DECREMENT,
                 spread if spread is not None else wx.CREDIT_GRID[3],
                 deal or wx.DEAL)


def main():
    print(f"\n{wx.DEAL.describe()}")
    print(f"{wx.INDEX_NAME} · spot {wx.SPOT:,.2f} · vol {wx.VOL:.0%} · "
          f"trade date {wx.TRADE_DATE}")

    section("1. CHOICE OF CREDIT SPREAD")
    print(f"{'Expected life':<16}{'Spread':>9}{'Fair value':>14}"
          f"{'Margin':>10}{'Verdict':>16}")
    for tenor in (1, 2, 3, 5):
        r = run(spread=wx.CREDIT_GRID[tenor])
        print(f"{tenor}Y{'':<15}{wx.CREDIT_GRID[tenor]*1e4:>7.0f}bp"
              f"{r.fair_value:>14.2f}{r.margin_pct*100:>9.2f}%"
              f"{'clears' if r.passes_floor else 'below floor':>16}")

    base = run()
    chosen = wx.nearest_grid_tenor(base.expected_life)
    print(f"\n  Simulated expected life: {base.expected_life:.2f} years")
    print(f"  Nearest grid line: {chosen}Y at "
          f"+{wx.CREDIT_GRID[chosen]*1e4:.0f} bp")
    print(f"  {wx.CREDIT_RATIONALE[chosen]}")
    print("  A wider spread would lower fair value and flatter the margin,")
    print("  so the line is taken from the model rather than chosen by eye.")

    section("2. BASE CASE")
    print(f"  Fair value                   {base.fair_value:>10,.2f}")
    print(f"  Gross structuring margin     {base.margin_pct*100:>10.2f} %")
    print(f"  Margin floor                 "
          f"{wx.DEAL.margin_floor*100:>10.2f} %")
    print(f"  Distance to floor            {base.margin_gap_bp:>10,.0f} bp")
    print()
    print(f"  Probability of early redemption  {base.prob_autocall:>8.1%}")
    print(f"  Probability of capital loss      "
          f"{base.prob_capital_loss:>8.1%}")
    print(f"  Coupons paid on average          "
          f"{base.avg_coupons_paid:>8.1f} of {len(wx.SCHEDULE)}")
    print(f"  PV coupons / PV redemption       "
          f"{base.pv_coupons:>8,.2f} / {base.pv_redemption:,.2f}")

    section("3. VOLATILITY REPRICING RISK")
    print(f"{'Vol':<8}{'Fair value':>14}{'Margin':>10}{'vs base':>14}")
    for v in (0.16, 0.17, 0.18):
        r = run(vol=v)
        tag = "  (base)" if abs(v - wx.VOL) < 1e-9 else ""
        print(f"{v*100:.0f}%{'':<5}{r.fair_value:>14.2f}"
              f"{r.margin_pct*100:>9.2f}%"
              f"{(r.margin_pct - base.margin_pct)*1e4:>12,.0f}bp{tag}")
    print("\n  The margin moves about a point of nominal per point of vol,")
    print("  which is more than the floor itself. Timing risk, not a lever.")

    section("4. REVISION LEVERS - ONE AT A TIME")
    print("  Breakeven = the value that brings the margin back to the floor\n")
    levers = [
        ("Coupon", "6.00% -> 5.75% p.a.",
         Deal(**{**wx.DEAL.__dict__, "coupon_rate_pa": 0.0575})),
        ("Coupon barrier", "60% -> 63.6%",
         Deal(**{**wx.DEAL.__dict__, "coupon_barrier": 0.636})),
        ("Capital barrier", "60% -> 62.1%",
         Deal(**{**wx.DEAL.__dict__, "capital_barrier": 0.621})),
    ]
    for name, move, d in levers:
        r = run(deal=d)
        print(f"  {name:<18}{move:<24}margin {r.margin_pct*100:5.2f} %  "
              f"loss prob {r.prob_capital_loss:5.1%}")
    print("\n  Lowering the coupon or capital barrier makes the note more")
    print("  generous and therefore more expensive: those levers cost")
    print("  margin. Raising them is what recovers it.")

    section("5. CONCLUSION")
    verdict = "clears" if base.passes_floor else "does not clear"
    print(f"  At the proposed terms the margin {verdict} the floor "
          f"({base.margin_pct:.2%} vs {wx.DEAL.margin_floor:.2%}).")
    if not base.passes_floor:
        print("  Launching unchanged is not available; one product lever")
        print("  has to move, and the volatility assumption is not one.")


if __name__ == "__main__":
    main()
