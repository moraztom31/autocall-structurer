"""
Phoenix autocallable pricer - Streamlit interface.

Two modes:

  Worked example  a single deal on a frozen snapshot, reproducible to
                  the cent, which is how a pricing committee works
  Note builder    pick an underlying, set the terms, price anything

    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import charts
import worked_example as wx
from deal import NOMINAL, Deal, build_schedule
from market import DEFAULT_RATE, DEFAULT_VOL, INDICES, fetch_quote, manual_quote
from pricer import payoff_at_maturity, price, simulate_paths

st.set_page_config(page_title="Phoenix Autocallable Pricer",
                   page_icon="◫", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2.4rem; max-width: 1240px;}
  [data-testid="stMetricValue"] {font-size: 1.75rem;}
  [data-testid="stMetricLabel"] {font-size: .8rem; letter-spacing: .02em;}
  h1 {font-weight: 650; letter-spacing: -.015em;}
  .lead {color: #5A6069; font-size: .95rem; line-height: 1.55;
         max-width: 62ch;}
  .note {color: #6B7079; font-size: .82rem; line-height: 1.5;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, ttl=900)
def cached_quote(ticker):
    return fetch_quote(ticker)


# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    st.markdown("### Mode")
    mode = st.radio(
        "Mode", ["Worked example", "Note builder"],
        label_visibility="collapsed",
        captions=["A frozen snapshot, priced exactly",
                  "Build and price your own note"],
    )

    st.divider()

    if mode == "Worked example":
        st.markdown("### Snapshot")
        st.caption(
            "Locked. A committee prices the reference snapshot agreed "
            "with trading, not a live feed."
        )
        st.markdown(
            f"<div class='note'>Underlying &nbsp;{wx.INDEX_NAME}<br>"
            f"Spot &nbsp;{wx.SPOT:,.2f}<br>"
            f"Implied vol &nbsp;{wx.VOL:.0%} (desk mid)<br>"
            f"Decrement &nbsp;{wx.DECREMENT:.0%} p.a.<br>"
            f"Trade date &nbsp;{wx.TRADE_DATE}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### Volatility scenario")
        vol = st.slider("Implied volatility", 0.12, 0.25, wx.VOL, 0.005,
                        format="%.3f",
                        help="Move this to size repricing risk, not to "
                             "clear the margin floor.")
        if abs(vol - wx.VOL) > 1e-9:
            st.warning(f"Off the desk mid of {wx.VOL:.0%} — this is a "
                       "sensitivity, not the base case.")

        st.markdown("### Credit spread")
        life_choice = st.select_slider(
            "Expected life", options=[1, 2, 3, 5], value=3,
            format_func=lambda x: f"{x}Y   +{wx.CREDIT_GRID[x]*1e4:.0f} bp",
        )
        st.caption(wx.CREDIT_RATIONALE[life_choice])

        st.markdown("### Terms")
        st.caption("One lever at a time, and never the volatility.")
        coupon = st.slider("Coupon p.a.", 0.03, 0.09,
                           wx.DEAL.coupon_rate_pa, 0.0025, format="%.4f")
        cpn_bar = st.slider("Coupon barrier", 0.40, 0.80,
                            wx.DEAL.coupon_barrier, 0.01, format="%.2f")
        ac_bar = st.slider("Autocall trigger", 0.85, 1.15,
                           wx.DEAL.autocall_barrier, 0.01, format="%.2f")
        cap_bar = st.slider("Capital barrier", 0.40, 0.80,
                            wx.DEAL.capital_barrier, 0.01, format="%.2f")
        issue = st.slider("Issue price", 0.95, 1.05,
                          wx.DEAL.issue_price, 0.005, format="%.3f")

        spot, decrement = wx.SPOT, wx.DECREMENT
        schedule = wx.SCHEDULE
        spread = wx.CREDIT_GRID[life_choice]
        underlying_name = wx.INDEX_NAME
        deal = Deal(maturity_years=5.0, coupon_rate_pa=coupon,
                    coupons_per_year=4, coupon_barrier=cpn_bar,
                    autocall_barrier=ac_bar, autocall_start_year=1.0,
                    autocalls_per_year=1, capital_barrier=cap_bar,
                    issue_price=issue, margin_floor=0.015)

    else:
        st.markdown("### Underlying")
        ticker = st.selectbox(
            "Index", list(INDICES), format_func=lambda t: INDICES[t])
        quote = cached_quote(ticker)

        if quote.is_live:
            st.success(f"{quote.name} · {quote.spot:,.2f} · {quote.as_of}")
            spot_default = quote.spot
        else:
            st.info("Live data unavailable — enter the level yourself.")
            spot_default = 5000.0

        spot = st.number_input("Spot level", min_value=1.0,
                               value=float(spot_default), step=10.0,
                               format="%.2f")

        st.markdown("### Volatility")
        rv = quote.realised_vol
        if rv:
            st.caption(f"1Y realised: **{rv:.1%}** — backward-looking, and "
                       "usually below the implied level an issuer trades on.")
        else:
            st.caption("No history available, so no realised estimate.")

        vol_source = st.radio(
            "Source", ["Enter implied volatility", "Use realised as a proxy"],
            label_visibility="collapsed",
            disabled=rv is None,
            index=0,
        )
        if vol_source.startswith("Use") and rv:
            vol = rv
            st.caption(f"Pricing at {vol:.1%} realised — a proxy, not a quote.")
        else:
            vol = st.number_input("Implied volatility", 0.05, 0.80,
                                  float(rv) if rv else DEFAULT_VOL,
                                  0.005, format="%.3f")

        st.markdown("### Curve and funding")
        flat_rate = st.number_input("Risk-free rate (cont. comp.)",
                                    -0.01, 0.12, DEFAULT_RATE, 0.0025,
                                    format="%.4f")
        spread = st.number_input("Issuer credit spread", 0.0, 0.05,
                                 0.0020, 0.0005, format="%.4f",
                                 help="Applied to the note's cash flows "
                                      "only, never to the index drift.")
        decrement = st.number_input("Index decrement p.a.", 0.0, 0.10,
                                    0.0, 0.005, format="%.3f",
                                    help="Set above zero to price a "
                                         "synthetic decrement version of "
                                         "this index.")

        st.markdown("### Terms")
        maturity = st.select_slider("Maturity (years)",
                                    options=[1, 2, 3, 4, 5, 6, 7, 8, 10],
                                    value=5)
        freq = st.select_slider("Coupon frequency", options=[1, 2, 4, 12],
                                value=4,
                                format_func=lambda f: {1: "Annual",
                                                       2: "Semi-annual",
                                                       4: "Quarterly",
                                                       12: "Monthly"}[f])
        coupon = st.slider("Coupon p.a.", 0.01, 0.20, 0.06, 0.0025,
                           format="%.4f")
        cpn_bar = st.slider("Coupon barrier", 0.30, 1.00, 0.60, 0.01,
                            format="%.2f")
        ac_bar = st.slider("Autocall trigger", 0.70, 1.20, 1.00, 0.01,
                           format="%.2f")
        ac_start = st.select_slider("First autocall after (years)",
                                    options=[0.5, 1.0, 1.5, 2.0, 3.0],
                                    value=1.0)
        ac_freq = st.select_slider("Autocall frequency",
                                   options=[1, 2, 4], value=1,
                                   format_func=lambda f: {1: "Annual",
                                                          2: "Semi-annual",
                                                          4: "Quarterly"}[f])
        cap_bar = st.slider("Capital barrier", 0.30, 1.00, 0.60, 0.01,
                            format="%.2f")
        issue = st.slider("Issue price", 0.90, 1.10, 1.00, 0.005,
                          format="%.3f")
        floor = st.slider("Margin floor", 0.0, 0.05, 0.015, 0.0025,
                          format="%.4f")

        deal = Deal(maturity_years=float(maturity), coupon_rate_pa=coupon,
                    coupons_per_year=freq, coupon_barrier=cpn_bar,
                    autocall_barrier=ac_bar, autocall_start_year=ac_start,
                    autocalls_per_year=ac_freq, capital_barrier=cap_bar,
                    issue_price=issue, margin_floor=floor)
        schedule = build_schedule(deal, flat_rate)
        underlying_name = quote.name

    st.divider()
    n_paths = st.select_slider("Simulated paths",
                               options=[10_000, 50_000, 200_000],
                               value=50_000,
                               format_func=lambda n: f"{n:,}")

# =====================================================================
# Price
# =====================================================================
paths = simulate_paths(schedule, spot, vol, decrement, n_paths)
res = price(schedule, spot, vol, decrement, spread, deal,
            n_paths=n_paths, paths=paths)

st.title("Phoenix Autocallable — Monte Carlo Pricer")
st.markdown(
    f"<p class='lead'>{deal.describe()} on {underlying_name}. "
    f"Fair value from {n_paths:,} simulated paths, and the gross "
    "structuring margin that follows from it.</p>",
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Fair value", f"{res.fair_value:,.2f}",
          help="What the promise costs the issuer to manufacture and hedge.")
m2.metric("Gross margin", f"{res.margin_pct:.2%}",
          f"{res.margin_gap_bp:+,.0f} bp vs floor",
          delta_color="normal" if res.passes_floor else "inverse",
          help="Issue price less fair value: what the issuer keeps before "
               "distribution fees.")
m3.metric("Margin floor", f"{deal.margin_floor:.2%}",
          help="Minimum the desk requires to cover distribution rebates, "
               "hedging slippage and model-to-market gaps.")
m4.metric("Expected life", f"{res.expected_life:.2f} y",
          help="Mean time to exit across simulated paths — autocall or "
               "maturity.")

if res.passes_floor:
    st.success(f"Margin clears the floor by {res.margin_gap_bp:,.0f} bp.")
else:
    st.error(
        f"Margin falls {abs(res.margin_gap_bp):,.0f} bp short. Revise a "
        "product lever — coupon, barrier, trigger, maturity or issue "
        "price — rather than the volatility assumption."
    )

tab_product, tab_pricing, tab_mc, tab_vol = st.tabs(
    ["The product", "Pricing", "Monte Carlo", "Volatility risk"])

# ---------------------------------------------------------------------
with tab_product:
    st.markdown(
        "<p class='lead'>A Phoenix autocallable pays a conditional coupon "
        "and can redeem early. Three tests, each on the index level "
        "relative to where it started.</p>",
        unsafe_allow_html=True,
    )

    st.dataframe(
        pd.DataFrame({
            "Test": ["Coupon", "Autocall", "Capital"],
            "When": [
                {1: "Annually", 2: "Semi-annually", 4: "Quarterly",
                 12: "Monthly"}[deal.coupons_per_year],
                {1: "Annually", 2: "Semi-annually",
                 4: "Quarterly"}[deal.autocalls_per_year]
                + f", from year {deal.autocall_start_year:g}",
                "At final valuation only",
            ],
            "Condition": [
                f"Index ≥ {deal.coupon_barrier:.0%} of initial",
                f"Index ≥ {deal.autocall_barrier:.0%} of initial",
                f"Index ≥ {deal.capital_barrier:.0%} of initial",
            ],
            "Outcome": [
                f"Pay {deal.coupon_amount:,.2f} per {NOMINAL:,.0f} note, "
                "otherwise nothing",
                "Redeem at par with the coupon due, and the note ends",
                "Redeem at par, otherwise par × index / initial",
            ],
        }),
        hide_index=True, use_container_width=True,
    )

    st.markdown(
        "<p class='note'>Coupons carry <b>no memory</b>: a missed coupon is "
        "not recovered later. A Phoenix keeps the coupon and autocall "
        "barriers apart, which is what lets it keep paying in a market "
        "that has fallen but not collapsed. An Athena aligns them and "
        "accrues coupons to the call date instead.</p>",
        unsafe_allow_html=True,
    )

    st.subheader("Redemption at maturity")
    st.markdown(
        "<p class='lead'>For a note that was never called — the only path "
        "on which the capital barrier is ever tested.</p>",
        unsafe_allow_html=True,
    )
    st.altair_chart(charts.payoff_curve(deal, payoff_at_maturity),
                    use_container_width=True)
    st.markdown(
        f"<p class='note'>The break at {deal.capital_barrier:.0%} is the "
        "whole risk of the structure. One tick above it the holder is "
        f"repaid in full; one tick below, redemption is "
        f"{deal.capital_barrier:.0%} of nominal and falls from there with "
        "the index. Describing this as protection down to "
        f"{1 - deal.capital_barrier:.0%} is accurate only on one day, at "
        "final valuation, and only if the note was never called.</p>",
        unsafe_allow_html=True,
    )

    with st.expander("Observation schedule"):
        st.dataframe(
            pd.DataFrame({
                "Observation": schedule.labels,
                "Years": np.round(schedule.times, 4),
                "Coupon test": "yes",
                "Autocall test": np.where(schedule.is_autocall, "yes", "—"),
                "Discount factor": np.round(schedule.df_ois, 5),
            }),
            hide_index=True, use_container_width=True, height=300,
        )

# ---------------------------------------------------------------------
with tab_pricing:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("What the holder experiences")
        st.dataframe(
            pd.DataFrame({
                "Measure": ["Probability of early redemption",
                            "Probability of capital loss",
                            "Coupons paid on average",
                            "Expected life"],
                "Value": [f"{res.prob_autocall:.1%}",
                          f"{res.prob_capital_loss:.1%}",
                          f"{res.avg_coupons_paid:.1f} of "
                          f"{len(schedule)}",
                          f"{res.expected_life:.2f} years"],
            }),
            hide_index=True, use_container_width=True,
        )
        st.markdown(
            "<p class='note'>The headline coupon is what a holder earns "
            "per year while the note lives, not what they collect over "
            "the full term: early redemption cuts the coupon stream "
            "short, and it fires in exactly the markets where it is "
            "pleasant to stay invested.</p>",
            unsafe_allow_html=True,
        )

    with right:
        st.subheader("Where the value sits")
        st.dataframe(
            pd.DataFrame({
                "Component": ["Coupons", "Redemption", "Fair value"],
                "Present value": [f"{res.pv_coupons:,.2f}",
                                  f"{res.pv_redemption:,.2f}",
                                  f"{res.fair_value:,.2f}"],
                "Share": [f"{res.pv_coupons/res.fair_value:.1%}",
                          f"{res.pv_redemption/res.fair_value:.1%}",
                          "100.0%"],
            }),
            hide_index=True, use_container_width=True,
        )
        st.markdown(
            "<p class='note'>Most of the value is the repayment of "
            "capital, not the income — which is the reverse of how these "
            "notes are usually discussed.</p>",
            unsafe_allow_html=True,
        )

    st.subheader("When the note leaves")
    st.altair_chart(charts.exit_distribution(res.exit_times, schedule),
                    use_container_width=True)

# ---------------------------------------------------------------------
with tab_mc:
    st.subheader("Simulated index paths")
    st.markdown(
        "<p class='lead'>Geometric Brownian motion, simulated exactly in "
        "log space, with antithetic sampling. Barriers are tested on "
        "observation dates only — what happens between them is not "
        "observed.</p>",
        unsafe_allow_html=True,
    )
    st.altair_chart(charts.path_fan(paths, schedule, spot, deal),
                    use_container_width=True)

    if decrement > 0:
        st.markdown(
            f"<p class='note'>The cloud tilts down because the index "
            f"deducts {decrement:.1%} a year by construction. Under the "
            "risk-neutral measure the drift is the risk-free rate less "
            "that deduction, so the underlying equities must outpace the "
            "deduction before the index goes anywhere — which is what "
            "makes the autocall harder to reach than it looks.</p>",
            unsafe_allow_html=True,
        )

    st.subheader("Distribution of outcomes")
    st.altair_chart(
        charts.payout_distribution(res.total_payout,
                                   deal.issue_price * NOMINAL),
        use_container_width=True)

    p = res.total_payout
    q1, q2, q3 = np.percentile(p, [5, 50, 95])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("5th percentile", f"{q1:,.0f}")
    c2.metric("Median", f"{q2:,.0f}")
    c3.metric("95th percentile", f"{q3:,.0f}")
    c4.metric("Paths below issue",
              f"{(p < deal.issue_price*NOMINAL).mean():.1%}")
    st.markdown(
        "<p class='note'>Undiscounted cash returned per note, coupons "
        "included. The shape is the point: a dense band of good outcomes "
        "and a thin, long tail of bad ones. An average return hides "
        "exactly that asymmetry.</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------
with tab_vol:
    st.subheader("Repricing risk between trade and issue")
    grid = np.round(np.arange(max(0.08, vol - 0.03),
                              vol + 0.0301, 0.005), 4)
    rows = []
    for v in grid:
        r = price(schedule, spot, v, decrement, spread, deal,
                  n_paths=n_paths)
        rows.append({"vol": float(v), "margin": r.margin_pct,
                     "fair_value": r.fair_value})

    st.altair_chart(charts.vol_sensitivity(rows, deal.margin_floor),
                    use_container_width=True)

    lo = next(r for r in rows if abs(r["vol"] - round(vol - 0.01, 4)) < 1e-6)
    hi = next(r for r in rows if abs(r["vol"] - round(vol + 0.01, 4)) < 1e-6)
    d_lo = (lo["margin"] - res.margin_pct) * 1e4
    d_hi = (hi["margin"] - res.margin_pct) * 1e4

    st.dataframe(
        pd.DataFrame({
            "Implied volatility": [f"{lo['vol']:.1%}", f"{vol:.1%}",
                                   f"{hi['vol']:.1%}"],
            "Fair value": [f"{lo['fair_value']:,.2f}",
                           f"{res.fair_value:,.2f}",
                           f"{hi['fair_value']:,.2f}"],
            "Gross margin": [f"{lo['margin']:.2%}", f"{res.margin_pct:.2%}",
                             f"{hi['margin']:.2%}"],
            "Versus base": [f"{d_lo:+,.0f} bp", "—", f"{d_hi:+,.0f} bp"],
        }),
        hide_index=True, use_container_width=True,
    )

    swing = (abs(d_lo) + abs(d_hi)) / 2
    st.info(
        f"One point of volatility moves the margin by about "
        f"{swing:,.0f} bp — against a floor of "
        f"{deal.margin_floor*1e4:,.0f} bp. The economics of the trade "
        "rest on an assumption that moves daily, which is an argument "
        "about execution timing, not a reason to pick the volatility "
        "that clears the floor."
    )

st.divider()
st.markdown(
    "<p class='note'>Flat Black volatility, no skew; lognormal, no jumps; "
    "a flat deterministic credit spread; no dynamic hedging cost, repo or "
    "market friction. Figures are indicative and not a quotation.</p>",
    unsafe_allow_html=True,
)
