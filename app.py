"""
Autocallable structuring tool - Streamlit interface.

Two modes:

  Worked example  one deal on a frozen snapshot, reproducible to the
                  cent, which is how a pricing committee works
  Note builder    pick an underlying, choose a structure, set the terms

    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import charts
import worked_example as wx
from deal import ATHENA, NOMINAL, PHOENIX, Deal, build_schedule
from market import DEFAULT_RATE, DEFAULT_VOL, INDICES, fetch_quote
from models import BlackParams, HestonParams, match_atm_vol
from pricer import payoff_at_maturity, price, simulate_paths

st.set_page_config(page_title="Autocallable Structurer",
                   page_icon="◫", layout="wide")

st.markdown("""
<style>
  /* Theme-neutral: backgrounds stay transparent so the app's own
     theme shows through, and borders and secondary text are drawn in
     a grey that recedes on white and on near-black alike. Nothing here
     needs to know which theme is active, so nothing flashes while the
     first run works it out. */
  .block-container {padding-top: 2rem; padding-bottom: 3rem;
                    max-width: 1280px;}
  h1 {font-weight: 640; letter-spacing: -.02em; font-size: 2.1rem;
      margin-bottom: .15rem;}
  h2 {font-weight: 600; letter-spacing: -.01em; font-size: 1.3rem;
      margin-top: .4rem;}
  h3 {font-weight: 600; font-size: 1.05rem;}

  .lead {font-size: .95rem; line-height: 1.6; max-width: 68ch;
         margin-bottom: .2rem; opacity: .78;}
  .note {font-size: .83rem; line-height: 1.55; max-width: 72ch;
         opacity: .62;}

  .tagrow {margin: .35rem 0 1.1rem 0;}
  .tag {display: inline-block; padding: .17rem .6rem; margin-right: .4rem;
        margin-bottom: .3rem; border-radius: 999px; font-size: .74rem;
        letter-spacing: .015em; color: #5091E6;
        background: rgba(80,145,230,.10);
        border: 1px solid rgba(80,145,230,.35);}

  .cards {display: flex; gap: .7rem; margin: .5rem 0 1rem 0;
          flex-wrap: wrap;}
  .card {flex: 1 1 0; min-width: 172px; padding: .85rem 1rem;
         border: 1px solid rgba(138,144,153,.32); border-radius: 10px;
         background: transparent;}
  .card .k {font-size: .73rem; text-transform: uppercase;
            letter-spacing: .055em; opacity: .58; margin-bottom: .3rem;}
  .card .v {font-size: 1.62rem; font-weight: 600; letter-spacing: -.02em;
            line-height: 1.12;}
  .card .s {font-size: .78rem; opacity: .58; margin-top: .2rem;}
  .card.good {border-color: rgba(80,145,230,.55);
              background: rgba(80,145,230,.08);}
  .card.bad  {border-color: rgba(220,112,73,.55);
              background: rgba(220,112,73,.08);}
  .card.good .v {color: #5091E6;}
  .card.bad  .v {color: #DC7049;}

  .ts {border: 1px solid rgba(138,144,153,.32); border-radius: 10px;
       padding: .2rem 1rem; background: transparent;}
  .ts table {width: 100%; border-collapse: collapse; font-size: .86rem;}
  .ts td {padding: .42rem 0;
          border-bottom: 1px solid rgba(138,144,153,.22);}
  .ts td:first-child {opacity: .58; width: 52%;}
  .ts tr:last-child td {border-bottom: none;}

  section[data-testid="stSidebar"] {
      border-right: 1px solid rgba(138,144,153,.28);}
  section[data-testid="stSidebar"] h3 {font-size: .78rem;
      text-transform: uppercase; letter-spacing: .07em; opacity: .58;
      margin-top: 1.1rem; margin-bottom: .1rem;}
  hr {margin: .9rem 0;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, ttl=900)
def cached_quote(ticker):
    return fetch_quote(ticker)


@st.cache_data(show_spinner=False)
def calibrated_heston(_schedule, spot, decrement, target_vol,
                      theta_v, kappa, xi, rho):
    """
    Heston scaled so its ATM implied volatility matches the Black input.

    Cached because the scaling itself runs several simulations, and
    without it every slider move would pay for the whole search again.
    """
    raw = HestonParams(v0=target_vol**2, theta=theta_v**2, kappa=kappa,
                       xi=xi, rho=rho)
    return match_atm_vol(_schedule, spot, raw, decrement, target_vol)


def cards(items):
    """Headline figures as a row of cards."""
    html = "<div class='cards'>"
    for label, value, sub, tone in items:
        cls = f"card {tone}".strip()
        sub = f"<div class='s'>{sub}</div>" if sub else ""
        html += (f"<div class='{cls}'><div class='k'>{label}</div>"
                 f"<div class='v'>{value}</div>{sub}</div>")
    st.markdown(html + "</div>", unsafe_allow_html=True)


def term_sheet(rows):
    body = "".join(f"<tr><td>{k}</td><td><b>{v}</b></td></tr>"
                   for k, v in rows)
    st.markdown(f"<div class='ts'><table>{body}</table></div>",
                unsafe_allow_html=True)


def lead(text):
    st.markdown(f"<p class='lead'>{text}</p>", unsafe_allow_html=True)


def note(text):
    st.markdown(f"<p class='note'>{text}</p>", unsafe_allow_html=True)


FREQ = {1: "Annual", 2: "Semi-annual", 4: "Quarterly", 12: "Monthly"}

# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    st.markdown("### Mode")
    mode = st.radio(
        "Mode", ["Worked example", "Note builder"],
        label_visibility="collapsed",
        captions=["A frozen snapshot, priced exactly",
                  "Build and price your own"],
    )

    if mode == "Worked example":
        st.markdown("### Snapshot")
        st.caption("Locked. A committee prices the reference snapshot "
                   "agreed with trading, never a live feed.")
        st.markdown(
            f"<p class='note'>{wx.INDEX_NAME}<br>"
            f"Spot {wx.SPOT:,.2f} · vol {wx.VOL:.0%} · "
            f"decrement {wx.DECREMENT:.0%} p.a.<br>"
            f"Trade date {wx.TRADE_DATE}</p>",
            unsafe_allow_html=True)

        st.markdown("### Volatility scenario")
        vol = st.slider("Implied volatility", 0.12, 0.25, wx.VOL, 0.005,
                        format="%.3f",
                        help="For sizing repricing risk - not a lever "
                             "for clearing the margin floor.")
        if abs(vol - wx.VOL) > 1e-9:
            st.warning(f"Off the desk mid of {wx.VOL:.0%}. This is a "
                       "sensitivity, not the base case.")

        st.markdown("### Credit spread")
        tenor = st.select_slider(
            "Expected life", options=[1, 2, 3, 5], value=3,
            format_func=lambda x: f"{x}Y   +{wx.CREDIT_GRID[x]*1e4:.0f} bp")
        st.caption(wx.CREDIT_RATIONALE[tenor])
        spread = wx.CREDIT_GRID[tenor]

        spot, decrement = wx.SPOT, wx.DECREMENT
        carry_label = "Index decrement"
        underlying = wx.INDEX_NAME
        maturity, freq = 5.0, 4
        ac_start, ac_freq = 1.0, 1
        flat_rate = None
    else:
        st.markdown("### Underlying")
        ticker = st.selectbox("Index", list(INDICES),
                              format_func=lambda t: INDICES[t])
        quote = cached_quote(ticker)
        if quote.is_live:
            st.success(f"{quote.spot:,.2f} · {quote.as_of}")
        else:
            st.info("Live data unavailable — enter the level below.")
        spot = st.number_input(
            "Spot level", min_value=1.0,
            value=float(quote.spot if quote.is_live else 5000.0),
            step=10.0, format="%.2f")

        st.markdown("### Volatility")
        rv = quote.realised_vol
        if rv:
            st.caption(f"1Y realised **{rv:.1%}** — backward-looking, and "
                       "usually below the implied level an issuer trades on.")
        else:
            st.caption("No history available, so no realised estimate.")
        use_rv = st.toggle("Use realised as a proxy", value=False,
                           disabled=rv is None)
        if use_rv and rv:
            vol = float(rv)
            st.caption(f"Pricing at {vol:.1%} realised — a proxy, not a quote.")
        else:
            vol = st.number_input("Implied volatility", 0.05, 0.80,
                                  float(rv) if rv else DEFAULT_VOL,
                                  0.005, format="%.3f")

        st.markdown("### Curve and funding")
        flat_rate = st.number_input("Risk-free rate (cont. comp.)",
                                    -0.01, 0.12, DEFAULT_RATE, 0.0025,
                                    format="%.4f")
        spread = st.number_input("Issuer credit spread", 0.0, 0.05, 0.0020,
                                 0.0005, format="%.4f",
                                 help="Applied to the note's cash flows "
                                      "only, never to the index drift.")
        st.markdown("### Carry")
        carry_kind = st.radio(
            "Underlying type", ["Standard index", "Decrement index"],
            label_visibility="collapsed",
            captions=["Pays dividends", "Fixed deduction instead"])
        if carry_kind == "Standard index":
            decrement = st.number_input(
                "Dividend yield p.a.", 0.0, 0.10, 0.030, 0.005,
                format="%.3f",
                help="Deducted from the risk-neutral drift. Leaving it "
                     "at zero overstates the forward and every figure "
                     "that follows from it.")
            carry_label = "Dividend yield"
        else:
            decrement = st.number_input(
                "Decrement p.a.", 0.0, 0.10, 0.050, 0.005,
                format="%.3f",
                help="A published deduction that replaces the dividend "
                     "yield rather than adding to it.")
            carry_label = "Index decrement"
        underlying = quote.name

        st.markdown("### Schedule")
        maturity = float(st.select_slider(
            "Maturity (years)", options=[1, 2, 3, 4, 5, 6, 7, 8, 10],
            value=5))
        freq = st.select_slider("Coupon frequency", options=[1, 2, 4, 12],
                                value=4, format_func=FREQ.get)
        ac_start = st.select_slider("First autocall after (years)",
                                    options=[0.5, 1.0, 1.5, 2.0, 3.0],
                                    value=1.0)
        ac_freq = st.select_slider("Autocall frequency", options=[1, 2, 4],
                                   value=1, format_func=FREQ.get)

    st.markdown("### Volatility model")
    model_name = st.radio(
        "Model", ["Black, flat", "Heston"], horizontal=True,
        label_visibility="collapsed",
        captions=["One sigma, no skew", "Stochastic vol, with skew"])

    if model_name == "Heston":
        st.caption("Parameterised, not calibrated — there is no option "
                   "surface here to fit, and a decrement index has "
                   "almost no listed options to fit to.")
        h_theta = st.slider("Long-run vol (theta)", 0.08, 0.40, 0.18,
                            0.01, format="%.2f")
        h_kappa = st.slider("Mean reversion (kappa)", 0.2, 5.0, 1.5, 0.1,
                            format="%.1f")
        h_xi = st.slider("Vol of vol (xi)", 0.05, 1.50, 0.55, 0.05,
                         format="%.2f",
                         help="Smile curvature.")
        h_rho = st.slider("Spot/vol correlation (rho)", -0.95, 0.0, -0.70,
                          0.05, format="%.2f",
                          help="Skew. Negative on equities: volatility "
                               "rises as the index falls.")
        match_atm = st.toggle(
            "Match ATM vol to the Black input", value=True,
            help="Scales v0 and theta so both models carry the same "
                 "at-the-money volatility. Without it, the gap between "
                 "them mixes the level of vol with the skew.")

    st.markdown("### Structure")
    style_label = st.radio(
        "Style", ["Phoenix", "Athena"], horizontal=True,
        label_visibility="collapsed",
        captions=["Separate barriers, coupons paid as due",
                  "One barrier, coupons accrue to redemption"])
    style = PHOENIX if style_label == "Phoenix" else ATHENA

    coupon = st.slider("Coupon p.a.", 0.01, 0.20, 0.06, 0.0025,
                       format="%.4f")
    if style == PHOENIX:
        cpn_bar = st.slider("Coupon barrier", 0.30, 1.00, 0.60, 0.01,
                            format="%.2f")
        memory = st.toggle("Memory coupons", value=False,
                           help="Missed coupons are recovered when the "
                                "barrier is next met.")
    else:
        cpn_bar, memory = 0.60, False
        st.caption("An Athena tests coupons at the autocall level and "
                   "pays the accrued stream on redemption.")

    ac_bar = st.slider("Autocall trigger", 0.70, 1.20, 1.00, 0.01,
                       format="%.2f")
    stepdown = st.slider("Trigger step-down p.a.", 0.0, 0.10, 0.0, 0.005,
                         format="%.3f",
                         help="Reduces the trigger each year, so the note "
                              "becomes progressively easier to call.")

    cap_bar = st.slider("Capital barrier", 0.30, 1.00, 0.60, 0.01,
                        format="%.2f")
    airbag = st.toggle("Airbag", value=False,
                       help="Below the barrier, loss is measured from a "
                            "strike rather than from the initial level.")
    air_k = st.slider("Airbag strike", 0.20, 1.00, float(cap_bar), 0.01,
                      format="%.2f", disabled=not airbag) if airbag \
        else cap_bar

    issue = st.slider("Issue price", 0.90, 1.10, 1.00, 0.005, format="%.3f")
    floor = (0.015 if mode == "Worked example"
             else st.slider("Margin floor", 0.0, 0.05, 0.015, 0.0025,
                            format="%.4f"))

    st.divider()
    n_paths = st.select_slider("Simulated paths",
                               options=[10_000, 50_000, 200_000],
                               value=50_000,
                               format_func=lambda n: f"{n:,}")

deal = Deal(style=style, maturity_years=maturity, coupon_rate_pa=coupon,
            coupons_per_year=freq, coupon_barrier=cpn_bar,
            coupon_memory=memory, autocall_barrier=ac_bar,
            autocall_start_year=ac_start, autocalls_per_year=ac_freq,
            autocall_stepdown_pa=stepdown, capital_barrier=cap_bar,
            airbag=airbag, airbag_strike=air_k, issue_price=issue,
            margin_floor=floor)

schedule = (wx.schedule_for(deal) if mode == "Worked example"
            else build_schedule(deal, flat_rate))

black = BlackParams(vol)
if model_name == "Heston":
    if match_atm:
        model = calibrated_heston(schedule, spot, decrement, vol,
                                  h_theta, h_kappa, h_xi, h_rho)
    else:
        model = HestonParams(v0=vol**2, theta=h_theta**2, kappa=h_kappa,
                             xi=h_xi, rho=h_rho)
else:
    model = black

# =====================================================================
# Price
# =====================================================================
paths = simulate_paths(schedule, spot, model, decrement, n_paths)
res = price(schedule, spot, model, decrement, spread, deal,
            n_paths=n_paths, paths=paths)

st.title("Autocallable Structurer")
lead(f"{deal.describe()} on {underlying}. Priced under "
     f"{model.describe()}, {n_paths:,} simulated paths, and the gross "
     "structuring margin that follows.")

tags = deal.features() + [model.label]
if tags:
    st.markdown("<div class='tagrow'>"
                + "".join(f"<span class='tag'>{t}</span>" for t in tags)
                + "</div>", unsafe_allow_html=True)

cards([
    ("Fair value", f"{res.fair_value:,.2f}", "cost to manufacture", ""),
    ("Gross margin", f"{res.margin_pct:.2%}",
     f"{res.margin_gap_bp:+,.0f} bp vs floor",
     "good" if res.passes_floor else "bad"),
    ("Margin floor", f"{deal.margin_floor:.2%}", "desk minimum", ""),
    ("Expected life", f"{res.expected_life:.2f} y", "mean time to exit", ""),
    ("Capital loss", f"{res.prob_capital_loss:.1%}",
     "probability at maturity", ""),
])

if not res.passes_floor:
    st.error(
        f"Margin falls {abs(res.margin_gap_bp):,.0f} bp short. Revise a "
        "product lever — coupon, barrier, trigger, maturity or issue "
        "price — rather than the volatility assumption.")

t_product, t_pricing, t_mc, t_vol, t_model = st.tabs(
    ["Structure", "Pricing", "Monte Carlo", "Volatility risk",
     "Model risk"])

# ---------------------------------------------------------------------
with t_product:
    c1, c2 = st.columns([1.05, 1])

    with c1:
        st.subheader("Term sheet")
        rows = [
            ("Style", "Athena" if deal.is_athena else "Phoenix"),
            ("Underlying", underlying),
            ("Maturity", f"{deal.maturity_years:g} years"),
            ("Issue price", f"{deal.issue_price:.1%}"),
            ("Coupon", f"{deal.coupon_rate_pa:.2%} p.a. "
                       f"({deal.coupon_amount:,.2f} per period)"),
            ("Coupon frequency", FREQ[deal.coupons_per_year]),
            ("Coupon condition",
             f"Index ≥ {deal.effective_coupon_barrier:.0%} of initial"),
            ("Coupon memory", "Yes" if deal.coupon_memory else "No"),
            ("Autocall trigger", f"{deal.autocall_barrier:.0%} of initial"
             + (f", less {deal.autocall_stepdown_pa:.1%} p.a."
                if deal.autocall_stepdown_pa else "")),
            ("Autocall dates",
             f"{FREQ[deal.autocalls_per_year]} from year "
             f"{deal.autocall_start_year:g}"),
            ("Capital barrier", f"{deal.capital_barrier:.0%} of initial"),
            ("Airbag", f"Struck at {deal.airbag_strike:.0%}"
             if deal.airbag else "None"),
        ]
        if decrement:
            rows.insert(2, (carry_label, f"{decrement:.2%} p.a."))
        term_sheet(rows)

    with c2:
        st.subheader("How it pays")
        st.dataframe(
            pd.DataFrame({
                "Test": ["Coupon", "Autocall", "Capital"],
                "When": [FREQ[deal.coupons_per_year],
                         f"{FREQ[deal.autocalls_per_year]}, from year "
                         f"{deal.autocall_start_year:g}",
                         "Final valuation only"],
                "Outcome": [
                    ("Accrued and paid on redemption"
                     if deal.is_athena else
                     ("Paid, with missed coupons recovered"
                      if deal.coupon_memory
                      else "Paid, otherwise lost for good")),
                    "Redeem at par with the coupon due; the note ends",
                    ("Par, otherwise the loss from the airbag strike"
                     if deal.airbag else
                     "Par, otherwise par × index / initial"),
                ],
            }),
            hide_index=True, use_container_width=True)

        if deal.is_athena:
            note("An Athena has a single barrier. Nothing is paid until "
                 "the note redeems, and then the whole accrued stream is "
                 "paid at once — so a holder who exits early collects "
                 "less in total but waits less for it.")
        elif deal.coupon_memory:
            note("With memory, a missed coupon is only deferred: the next "
                 "date on which the barrier is met pays the backlog too. "
                 "That is worth real money, which is why it has to be "
                 "funded from somewhere else in the terms.")
        else:
            note("Coupons carry no memory: a missed coupon is never "
                 "recovered. A Phoenix keeps its coupon barrier below the "
                 "autocall level, which is what lets it keep paying in a "
                 "market that has fallen but not collapsed.")

    st.subheader("Redemption at maturity")
    lead("For a note that was never called — the only path on which the "
         "capital barrier is ever tested.")
    st.altair_chart(
        charts.payoff_curve(deal, payoff_at_maturity, len(schedule)),
        use_container_width=True)

    if deal.airbag and deal.airbag_strike <= deal.capital_barrier + 1e-9:
        note(f"The airbag struck at {deal.airbag_strike:.0%} closes the "
             "cliff entirely: redemption runs continuously through the "
             "barrier. That is the most expensive thing this structure "
             "can buy, because the cliff is precisely what the coupon is "
             "paid for.")
    else:
        note(f"The break at {deal.capital_barrier:.0%} is the whole risk "
             "of the structure. One tick above it the holder is repaid in "
             f"full; one tick below, redemption drops to "
             f"{deal.capital_barrier:.0%} of nominal and falls from there "
             "with the index. Calling this protection down to "
             f"{1 - deal.capital_barrier:.0%} is accurate on one day only, "
             "at final valuation, and only if the note was never called.")

    if deal.autocall_stepdown_pa:
        st.subheader("Autocall trigger schedule")
        st.altair_chart(charts.stepdown_schedule(schedule),
                        use_container_width=True)
        note("A step-down makes each successive call easier than the last. "
             "It shortens expected life, which cuts the issuer's exposure "
             "— so unusually among the levers, it can improve margin and "
             "the client's odds of an early exit at the same time.")

    with st.expander("Observation schedule"):
        st.dataframe(
            pd.DataFrame({
                "Observation": schedule.labels,
                "Years": np.round(schedule.times, 4),
                "Coupon test": "yes",
                "Autocall trigger": [
                    "—" if np.isnan(x) else f"{x:.1%}"
                    for x in schedule.autocall_level],
                "Discount factor": np.round(schedule.df_ois, 5),
            }),
            hide_index=True, use_container_width=True, height=300)

# ---------------------------------------------------------------------
with t_pricing:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("What the holder experiences")
        st.dataframe(
            pd.DataFrame({
                "Measure": ["Probability of early redemption",
                            "Probability of capital loss",
                            "Coupons paid on average",
                            "Expected life"],
                "Value": [f"{res.prob_autocall:.1%}",
                          f"{res.prob_capital_loss:.1%}",
                          f"{res.avg_coupons_paid:.1f} of {len(schedule)}",
                          f"{res.expected_life:.2f} years"],
            }),
            hide_index=True, use_container_width=True)
        note("The headline coupon is what a holder earns per year while "
             "the note lives, not what they collect over the full term: "
             "early redemption cuts the stream short, and it fires in "
             "exactly the markets where staying invested would have been "
             "pleasant.")
    with c2:
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
            hide_index=True, use_container_width=True)
        note("Most of the value is the repayment of capital, not the "
             "income — the reverse of how these notes are usually "
             "discussed.")

    st.subheader("When the note leaves")
    st.altair_chart(charts.exit_distribution(res.exit_times, schedule),
                    use_container_width=True)

# ---------------------------------------------------------------------
with t_mc:
    st.subheader("Simulated index paths")
    lead("Geometric Brownian motion, simulated exactly in log space, with "
         "antithetic sampling. Barriers are tested on observation dates "
         "only — what happens between them is not observed.")
    st.altair_chart(charts.path_fan(paths, schedule, spot, deal),
                    use_container_width=True)

    if decrement > 0 and carry_label == "Index decrement":
        note(f"The cloud tilts down because the index deducts "
             f"{decrement:.1%} a year by construction. Under the "
             "risk-neutral measure the drift is the risk-free rate less "
             "that deduction, so the underlying equities have to outpace "
             "the deduction before the index goes anywhere — which is "
             "what makes the autocall harder to reach than it looks.")
    elif decrement > 0:
        note(f"The drift is the risk-free rate less the "
             f"{decrement:.1%} dividend yield. A price index earns no "
             "dividends, so under the risk-neutral measure it drifts "
             "below the funding rate — and the holder of the note "
             "receives none of them either.")

    st.subheader("Distribution of outcomes")
    st.altair_chart(
        charts.payout_distribution(res.total_payout,
                                   deal.issue_price * NOMINAL),
        use_container_width=True)

    p = res.total_payout
    q5, q50, q95 = np.percentile(p, [5, 50, 95])
    cards([
        ("5th percentile", f"{q5:,.0f}", "", ""),
        ("Median", f"{q50:,.0f}", "", ""),
        ("95th percentile", f"{q95:,.0f}", "", ""),
        ("Below issue",
         f"{(p < deal.issue_price*NOMINAL).mean():.1%}", "of paths", ""),
    ])
    note("Undiscounted cash returned per note, coupons included. The "
         "shape is the point: a dense band of good outcomes and a thin, "
         "long tail of bad ones. An average return hides exactly that "
         "asymmetry.")

# ---------------------------------------------------------------------
with t_vol:
    st.subheader("Repricing risk between trade and issue")
    step = 0.005 if model_name == "Black, flat" else 0.01
    grid = np.round(np.arange(max(0.06, vol - 0.03), vol + 0.0301, step), 4)
    sweep_paths = n_paths if model_name == "Black, flat" else min(n_paths,
                                                                 20_000)

    def shifted(v):
        """Same model, moved to a new volatility level."""
        if model_name == "Black, flat":
            return BlackParams(v)
        k = (v / vol) ** 2
        return HestonParams(v0=model.v0 * k, theta=model.theta * k,
                            kappa=model.kappa,
                            xi=model.xi * float(np.sqrt(k)), rho=model.rho)

    rows = []
    for v in grid:
        r = price(schedule, spot, shifted(float(v)), decrement, spread,
                  deal, n_paths=sweep_paths)
        rows.append({"vol": float(v), "margin": r.margin_pct,
                     "fair_value": r.fair_value})

    st.altair_chart(charts.vol_sensitivity(rows, deal.margin_floor),
                    use_container_width=True)

    def at(v):
        return min(rows, key=lambda r: abs(r["vol"] - v))

    lo, hi = at(vol - 0.01), at(vol + 0.01)
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
        hide_index=True, use_container_width=True)

    swing = (abs(d_lo) + abs(d_hi)) / 2
    st.info(f"One point of volatility moves the margin by about "
            f"{swing:,.0f} bp, against a floor of "
            f"{deal.margin_floor*1e4:,.0f} bp. The economics rest on an "
            "assumption that moves daily — an argument about execution "
            "timing, not a reason to adopt whichever volatility clears "
            "the floor.")

# ---------------------------------------------------------------------
with t_model:
    st.subheader("What the volatility model is worth")
    lead("An autocallable holder is short a down-and-in put struck at "
         "the capital barrier, so the price is largely a statement "
         "about the left tail. A flat volatility gives every strike the "
         "same number and has no skew at all — which is why the model "
         "choice is not a detail here.")

    with st.spinner("Pricing under both models…"):
        cmp_paths = min(n_paths, 50_000)
        rb = price(schedule, spot, black, decrement, spread, deal,
                   n_paths=cmp_paths)
        h_ref = calibrated_heston(schedule, spot, decrement, vol,
                                  0.18, 1.5, 0.55, -0.70)
        rh = price(schedule, spot, h_ref, decrement, spread, deal,
                   n_paths=cmp_paths)

    cards([
        ("Black, flat", f"{rb.margin_pct:.2%}", "no skew", ""),
        ("Heston", f"{rh.margin_pct:.2%}", "same ATM vol", ""),
        ("Model gap", f"{(rh.margin_pct - rb.margin_pct)*1e4:+,.0f} bp",
         "skew and curvature", ""),
        ("Margin floor", f"{deal.margin_floor:.2%}", "desk minimum", ""),
    ])

    st.dataframe(
        pd.DataFrame({
            "Model": ["Black, flat", "Heston, ATM-matched"],
            "Fair value": [f"{rb.fair_value:,.2f}", f"{rh.fair_value:,.2f}"],
            "Gross margin": [f"{rb.margin_pct:.2%}", f"{rh.margin_pct:.2%}"],
            "Early redemption": [f"{rb.prob_autocall:.1%}",
                                 f"{rh.prob_autocall:.1%}"],
            "Capital loss": [f"{rb.prob_capital_loss:.1%}",
                             f"{rh.prob_capital_loss:.1%}"],
            "Coupons paid": [f"{rb.avg_coupons_paid:.1f}",
                             f"{rh.avg_coupons_paid:.1f}"],
        }),
        hide_index=True, use_container_width=True)

    note("Both models are set to the same at-the-money volatility. "
         "Without that step the comparison would mix two different "
         "things — the level of volatility and the shape of the smile — "
         "and only the second is the question.")

    st.subheader("Where the gap comes from")
    lead("Correlation between spot and volatility is what creates skew. "
         "Sweeping it, with the ATM level held fixed, separates the "
         "skew effect from the curvature that vol-of-vol alone produces.")

    with st.spinner("Sweeping correlation…"):
        sweep = []
        for rho in (0.0, -0.3, -0.5, -0.7, -0.9):
            p = calibrated_heston(schedule, spot, decrement, vol,
                                  0.18, 1.5, 0.55, rho)
            r = price(schedule, spot, p, decrement, spread, deal,
                      n_paths=min(n_paths, 20_000))
            sweep.append({"rho": rho, "margin": r.margin_pct,
                          "gap": (r.margin_pct - rb.margin_pct) * 1e4,
                          "call": r.prob_autocall,
                          "loss": r.prob_capital_loss})

    st.dataframe(
        pd.DataFrame({
            "Correlation": [f"{r['rho']:+.1f}" for r in sweep],
            "Gross margin": [f"{r['margin']:.2%}" for r in sweep],
            "Versus flat Black": [f"{r['gap']:+,.0f} bp" for r in sweep],
            "Early redemption": [f"{r['call']:.1%}" for r in sweep],
            "Capital loss": [f"{r['loss']:.1%}" for r in sweep],
        }),
        hide_index=True, use_container_width=True)

    st.warning(
        "Read this as model risk, not as a better price. None of these "
        "correlations is calibrated to anything — there is no surface "
        "in this project to fit. The finding is the size of the spread: "
        "the model moves the margin by more than the floor itself, and "
        "the direction depends on a parameter the flat model never "
        "makes you state."
    )

    note("Production autocall books go further again, to "
         "local-stochastic volatility. Local vol alone reprices today's "
         "smile but misstates forward skew; stochastic vol alone does "
         "not reprice the smile exactly. Heston is the step that makes "
         "the question visible, not the end of it.")

st.divider()
note("Lognormal or Heston dynamics, no jumps; Heston parameterised "
     "rather than calibrated; a flat deterministic credit spread; no "
     "dynamic hedging cost, repo or market friction. Figures are "
     "indicative and not a quotation.")
