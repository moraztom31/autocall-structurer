"""
Interface Streamlit pour le pricer Phoenix autocallable.

    streamlit run app.py

Le cas de base est toujours affiche a cote des valeurs modifiees :
les curseurs servent a l'analyse de sensibilite, pas a faire passer
la marge en bougeant les hypotheses de marche.
"""

import pandas as pd
import streamlit as st

from market_data import (
    CREDIT_GRID,
    DATES,
    MARGIN_FLOOR,
    S0,
    VOL_BASE,
)
from pricer import price, simulate_paths

st.set_page_config(page_title="Phoenix Autocall Pricer", layout="wide")

st.title("Phoenix Autocallable — Monte Carlo Pricer")
st.caption(
    f"Indice decrement 5 % · S₀ = {S0:,.2f} · snapshot fige · "
    "50 000 chemins, variables antithetiques"
)

# ---------------------------------------------------------------------
# Barre laterale
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Hypotheses de marche")
    st.caption("Lockees par le cas. A bouger pour le risque de repricing uniquement.")
    vol = st.slider("Volatilite implicite", 0.12, 0.25, VOL_BASE, 0.005,
                    format="%.3f")
    if abs(vol - VOL_BASE) > 1e-9:
        st.warning(f"Hors mid desk ({VOL_BASE:.0%}) — sensibilite, pas cas de base.")

    expected_life = st.select_slider(
        "Duree de vie attendue (grille ALM)",
        options=[1, 2, 3, 5],
        value=3,
        format_func=lambda x: f"{x}Y  (+{CREDIT_GRID[x]*1e4:.0f} bp)",
    )

    st.divider()
    st.header("Termes du produit")
    st.caption("Leviers de revision — un seul a la fois (Option B).")
    coupon = st.slider("Coupon annuel", 0.03, 0.08, 0.06, 0.0025, format="%.4f")
    cpn_bar = st.slider("Barriere coupon", 0.40, 0.80, 0.60, 0.01, format="%.2f")
    ac_bar = st.slider("Trigger autocall", 0.85, 1.15, 1.00, 0.01, format="%.2f")
    cap_bar = st.slider("Barriere capitale", 0.40, 0.80, 0.60, 0.01, format="%.2f")
    issue = st.slider("Prix d'emission", 0.95, 1.05, 1.00, 0.005, format="%.3f")

    st.divider()
    n_paths = st.select_slider("Nombre de chemins",
                               options=[10_000, 50_000, 100_000], value=50_000)

# ---------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------
kwargs = dict(
    n_paths=n_paths, vol=vol, expected_life=expected_life,
    coupon_rate=coupon / 4, coupon_barrier=cpn_bar,
    autocall_barrier=ac_bar, capital_barrier=cap_bar, issue_price=issue,
)
res = price(**kwargs)
base = price(n_paths=n_paths, expected_life=3)

# ---------------------------------------------------------------------
# Resultat principal
# ---------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Fair value", f"{res.fair_value:,.2f} €",
          f"{res.fair_value - base.fair_value:+.2f} vs base")
c2.metric("Marge brute", f"{res.margin_pct:.2%}",
          f"{(res.margin_pct - base.margin_pct)*1e4:+.0f} bp vs base")
c3.metric("Plancher desk", f"{MARGIN_FLOOR:.2%}")
c4.metric("Verdict", "PASSE" if res.passes_floor else "SOUS PLANCHER")

if res.passes_floor:
    st.success(
        f"Marge de {res.margin_pct:.2%}, soit "
        f"{(res.margin_pct - MARGIN_FLOOR)*1e4:.0f} bp au-dessus du plancher."
    )
else:
    st.error(
        f"Marge de {res.margin_pct:.2%}. Il manque "
        f"{(MARGIN_FLOOR - res.margin_pct)*1e4:.0f} bp — reviser un levier produit "
        "(pas l'hypothese de volatilite)."
    )

# ---------------------------------------------------------------------
# Onglets
# ---------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Diagnostics", "Sensibilite vol", "Chemins simules"])

with tab1:
    left, right = st.columns(2)
    with left:
        st.subheader("Ce que vit le client")
        st.dataframe(
            pd.DataFrame({
                "Mesure": ["Duree de vie moyenne", "Proba de rappel anticipe",
                           "Proba de perte en capital", "Coupons payes (moyenne)"],
                "Valeur": [f"{res.expected_life:.2f} ans",
                           f"{res.prob_autocall:.1%}",
                           f"{res.prob_capital_loss:.1%}",
                           f"{res.avg_coupons_paid:.1f} / 20"],
            }),
            hide_index=True, use_container_width=True,
        )
        st.caption(
            "La duree de vie simulee justifie le choix de ligne dans la grille ALM."
        )
    with right:
        st.subheader("Decomposition de la fair value")
        st.dataframe(
            pd.DataFrame({
                "Composante": ["PV des coupons", "PV du remboursement", "Total"],
                "EUR": [f"{res.pv_coupons:,.2f}", f"{res.pv_redemption:,.2f}",
                        f"{res.fair_value:,.2f}"],
                "% FV": [f"{res.pv_coupons/res.fair_value:.1%}",
                         f"{res.pv_redemption/res.fair_value:.1%}", "100.0%"],
            }),
            hide_index=True, use_container_width=True,
        )

with tab2:
    st.subheader("Risque de repricing avant emission")
    rows = []
    for v in (0.16, 0.17, 0.18):
        r = price(n_paths=n_paths, vol=v, expected_life=expected_life,
                  coupon_rate=coupon / 4, coupon_barrier=cpn_bar,
                  autocall_barrier=ac_bar, capital_barrier=cap_bar,
                  issue_price=issue)
        rows.append({
            "Volatilite": f"{v:.0%}",
            "Fair value": round(r.fair_value, 2),
            "Marge": f"{r.margin_pct:.2%}",
            "Passe le plancher": "oui" if r.passes_floor else "non",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.info(
        "La marge bouge d'environ 1 point pour 1 point de volatilite — plus que "
        "le plancher lui-meme. Si la vol se reprice avant l'emission, l'economie "
        "du deal change entierement. C'est un argument de calendrier, pas une "
        "invitation a choisir la vol qui arrange."
    )

with tab3:
    st.subheader("Trajectoires de l'indice")
    paths = simulate_paths(400, vol)
    df = pd.DataFrame(paths[:120].T, index=DATES)
    st.line_chart(df, height=360)
    st.caption(
        f"120 chemins. Barriere coupon a {cpn_bar*S0:,.0f} · "
        f"trigger autocall a {ac_bar*S0:,.0f}. Le decrement de 5 % p.a. "
        "tire la distribution vers le bas : le drift risque-neutre est negatif."
    )
