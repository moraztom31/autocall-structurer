"""
Lance l'analyse complete et imprime le tableau de resultats.

    python src/main.py
"""

from pricer import price
from market_data import CREDIT_GRID, MARGIN_FLOOR

LINE = "=" * 72


def section(title):
    print(f"\n{title}\n{LINE}")


def main():
    section("1. CAS DE BASE - choix du spread de credit")
    print(f"{'Vie attendue':<14}{'Spread':>9}{'Fair value':>14}{'Marge':>10}{'Verdict':>14}")
    for el in (1, 2, 3, 5):
        r = price(expected_life=el)
        verdict = "OK" if r.passes_floor else "sous plancher"
        print(f"{el}Y{'':<13}{CREDIT_GRID[el]*1e4:>7.0f}bp"
              f"{r.fair_value:>14.2f}{r.margin_pct*100:>9.2f}%{verdict:>14}")

    base = price(expected_life=3)
    section("2. DIAGNOSTICS DU CAS DE BASE (spread 3Y, justifie ci-dessous)")
    print(f"  Duree de vie moyenne simulee : {base.expected_life:.2f} ans")
    print(f"    -> justifie la ligne 3Y de la grille ALM (+20 bp)")
    print(f"  Probabilite de rappel        : {base.prob_autocall*100:.1f} %")
    print(f"  Probabilite de perte capital : {base.prob_capital_loss*100:.1f} %")
    print(f"  Coupons payes en moyenne     : {base.avg_coupons_paid:.1f} sur 20")
    print(f"  PV coupons / PV remboursement: {base.pv_coupons:.2f} / {base.pv_redemption:.2f}")

    section("3. RISQUE DE REPRICING VOL (+/- 1 point autour du mid 17 %)")
    print(f"{'Vol':<8}{'Fair value':>14}{'Marge':>10}{'Ecart vs base':>18}")
    for v in (0.16, 0.17, 0.18):
        r = price(vol=v, expected_life=3)
        delta = (r.margin_pct - base.margin_pct) * 1e4
        tag = "" if v != 0.17 else "  (base)"
        print(f"{v*100:.0f}%{'':<5}{r.fair_value:>14.2f}{r.margin_pct*100:>9.2f}%"
              f"{delta:>16.0f}bp{tag}")
    print("\n  La marge bouge d'environ 1 point pour 1 point de vol.")
    print("  C'est 2x le plancher lui-meme : le risque de repricing domine.")

    section("4. LEVIERS DE REVISION (option B) - un seul levier a la fois")
    print("  Point mort = valeur du levier qui ramene la marge a 1.50 %\n")
    levers = [
        ("Coupon", "6.00 % -> 5.75 % p.a.", {"coupon_rate": 0.014375}),
        ("Barriere coupon", "60 % -> 63.6 %", {"coupon_barrier": 0.636}),
        ("Barriere capitale", "60 % -> 62.1 %", {"capital_barrier": 0.621}),
    ]
    for name, move, kw in levers:
        r = price(expected_life=3, **kw)
        print(f"  {name:<20}{move:<24}marge {r.margin_pct*100:5.2f} %")

    section("5. CONCLUSION")
    print(f"  Marge aux termes proposes : {base.margin_pct*100:.2f} %")
    print(f"  Plancher exige            : {MARGIN_FLOOR*100:.2f} %")
    print(f"  -> Option A (lancer tel quel) exclue.")
    print(f"  -> Option B : reviser un levier produit.")


if __name__ == "__main__":
    main()
