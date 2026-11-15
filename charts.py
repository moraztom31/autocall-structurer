"""
Charts for the pricer interface.

House rules kept throughout: one hue per series, no second y-axis, a
legend whenever more than one series is drawn, recessive grid lines,
and a tooltip on every mark. Colour carries meaning only where the
meaning is real - capital at risk against capital returned - and is
always backed by a label, never left to carry identity on its own.

ONE palette, not one per theme. The app ships dark, but a visitor can
switch at any time and Streamlit does not report the active theme
reliably on the first run - detecting it meant the page rendered in the
wrong palette for a moment before correcting itself. So the two hues
are chosen to clear the full set of checks against a white surface and
a near-black one alike, and the chrome around them is drawn in neutral
grey at low opacity, which recedes on either background. Nothing has to
know which theme is on.
"""

import altair as alt
import numpy as np
import pandas as pd

PROTECTED = "#5091E6"     # capital returned in full
AT_RISK = "#DC7049"       # capital reduced with the index
NEUTRAL = "#8A9099"       # rules and reference marks
LABEL = "#8A9099"         # axis labels and titles


def _axis():
    return alt.Axis(grid=True, gridColor=NEUTRAL, gridOpacity=0.22,
                    domainColor=NEUTRAL, domainOpacity=0.35,
                    tickColor=NEUTRAL, tickOpacity=0.35,
                    labelColor=LABEL, titleColor=LABEL,
                    labelFontSize=11, titleFontSize=12)


def _axis():
    return alt.Axis(grid=True, gridColor=NEUTRAL, gridOpacity=0.8,
                    domainColor=NEUTRAL, tickColor=NEUTRAL,
                    labelColor=LABEL, titleColor=LABEL,
                    labelFontSize=11, titleFontSize=12)


def _finish(chart, height, legend=None):
    out = chart.properties(height=height).configure_view(strokeWidth=0)
    if legend:
        out = out.configure_legend(labelColor=LABEL, titleColor=LABEL,
                                   labelFontSize=11, titleFontSize=11,
                                   **legend)
    return out


def payoff_curve(deal, payoff_fn, n_obs=None, height=340):
    """
    Redemption at final valuation against the final index level.

    Drawn for a note that was never called, which is the only path on
    which the capital barrier is ever tested.

    The two sides of the barrier are drawn as separate segments. Without
    an airbag they do not meet, and that gap is the point: the drop is a
    cliff, not a slope, and a line joining them would draw a transition
    that does not exist. With an airbag struck at or below the barrier
    the segments meet on their own, which is exactly what the feature
    buys.
    """
    barrier = deal.capital_barrier

    below = np.linspace(0.0, barrier - 1e-6, 180)
    above = np.linspace(barrier, 1.60, 180)

    df = pd.concat([
        pd.DataFrame({"level": below,
                      "payout": payoff_fn(deal, below, n_obs),
                      "region": "Capital at risk"}),
        pd.DataFrame({"level": above,
                      "payout": payoff_fn(deal, above, n_obs),
                      "region": "Capital returned"}),
    ])
    df["level_pct"] = df["level"] * 100
    y_top = float(df["payout"].max()) * 1.12

    line = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("level_pct:Q", title="Final index level (% of initial)",
                scale=alt.Scale(domain=[0, 160]), axis=_axis()),
        y=alt.Y("payout:Q", title="Redemption per 1,000 note",
                scale=alt.Scale(domain=[0, y_top]), axis=_axis()),
        color=alt.Color("region:N", title=None,
                        scale=alt.Scale(domain=["Capital at risk",
                                                "Capital returned"],
                                        range=[AT_RISK,
                                               PROTECTED])),
        detail="region:N",
        tooltip=[alt.Tooltip("level_pct:Q", title="Index", format=".0f"),
                 alt.Tooltip("payout:Q", title="Redemption", format=",.0f"),
                 alt.Tooltip("region:N", title="")],
    )

    rule = alt.Chart(pd.DataFrame({"x": [barrier * 100]})).mark_rule(
        color=NEUTRAL, strokeDash=[5, 4], strokeWidth=1.5
    ).encode(x="x:Q")

    label = alt.Chart(pd.DataFrame({
        "x": [barrier * 100], "y": [y_top * 0.96],
        "t": [f"Capital barrier {barrier:.0%}"],
    })).mark_text(align="left", dx=7, fontSize=11,
                  color=NEUTRAL).encode(x="x:Q", y="y:Q", text="t:N")

    return _finish(rule + line + label, height,
                   legend={"orient": "top-left", "symbolType": "stroke",
                           "symbolStrokeWidth": 3})


def path_fan(paths, schedule, spot, deal, n_shown=140, height=360):
    """
    A sample of simulated index paths, with the two barriers marked.

    Thin, heavily transparent lines: the shape of the cloud is the point,
    not any single trajectory. The downward tilt is the decrement.
    """
    n_shown = min(n_shown, paths.shape[0])
    sample = paths[:n_shown]

    df = pd.DataFrame({
        "t": np.tile(schedule.times, n_shown),
        "level": sample.ravel(),
        "path": np.repeat(np.arange(n_shown), len(schedule)),
    })

    lines = alt.Chart(df).mark_line(
        strokeWidth=0.6, opacity=0.26, color=PROTECTED
    ).encode(
        x=alt.X("t:Q", title="Years from trade date", axis=_axis()),
        y=alt.Y("level:Q", title="Index level", axis=_axis()),
        detail="path:N",
    )

    marks = pd.DataFrame({
        "y": [deal.autocall_barrier * spot, deal.coupon_barrier * spot],
        "label": [f"Autocall {deal.autocall_barrier:.0%}",
                  f"Coupon / capital {deal.coupon_barrier:.0%}"],
    })
    rules = alt.Chart(marks).mark_rule(
        strokeDash=[5, 4], strokeWidth=1.5, color=NEUTRAL
    ).encode(y="y:Q", tooltip=["label:N"])

    text = alt.Chart(marks).mark_text(
        align="left", dx=6, dy=-7, fontSize=11, color=NEUTRAL
    ).encode(y="y:Q", text="label:N", x=alt.value(8))

    return _finish(lines + rules + text, height)


def payout_distribution(total_payout, issue_amount=1000.0, height=320):
    """
    Undiscounted cash returned per note across simulated paths.

    The shape matters more than any summary statistic: a Phoenix pays a
    narrow band of good outcomes very often and a long left tail rarely,
    which is exactly what an average return conceals.
    """
    df = pd.DataFrame({"payout": total_payout})
    df["region"] = np.where(df["payout"] < issue_amount,
                            "Below issue price", "At or above issue price")

    chart = alt.Chart(df).mark_bar(opacity=0.9).encode(
        x=alt.X("payout:Q", bin=alt.Bin(maxbins=70),
                title="Total cash returned per 1,000 note", axis=_axis()),
        y=alt.Y("count():Q", title="Simulated paths", axis=_axis()),
        color=alt.Color("region:N", title=None,
                        scale=alt.Scale(
                            domain=["Below issue price",
                                    "At or above issue price"],
                            range=[AT_RISK, PROTECTED])),
        tooltip=[alt.Tooltip("count():Q", title="Paths"),
                 alt.Tooltip("region:N", title="")],
    )
    return _finish(chart, height, legend={"orient": "top-right"})


def exit_distribution(exit_times, schedule, height=300):
    """
    When the note leaves: each autocall date, then scheduled maturity.

    The last bar lumps together every path that ran the full term, called
    or not, so it is always the tallest when calls are unlikely.
    """
    uniq, counts = np.unique(np.round(exit_times, 4), return_counts=True)
    share = counts / counts.sum()

    labels = []
    for t in uniq:
        idx = int(np.argmin(np.abs(schedule.times - t)))
        tag = "maturity" if idx == len(schedule) - 1 else "autocall"
        labels.append(f"{schedule.labels[idx]}  ({tag})")

    df = pd.DataFrame({"when": labels, "share": share, "paths": counts})

    x_axis = alt.Axis(format="%", grid=True, gridColor=NEUTRAL,
                      gridOpacity=0.22, domainColor=NEUTRAL,
                      domainOpacity=0.35, tickColor=NEUTRAL,
                      tickOpacity=0.35, labelColor=LABEL,
                      titleColor=LABEL, labelFontSize=11,
                      titleFontSize=12)

    bars = alt.Chart(df).mark_bar(
        color=PROTECTED, cornerRadiusEnd=4, size=26
    ).encode(
        y=alt.Y("when:N", title=None, sort=None, axis=_axis()),
        x=alt.X("share:Q", title="Share of simulated paths", axis=x_axis),
        tooltip=[alt.Tooltip("when:N", title="Exit"),
                 alt.Tooltip("share:Q", title="Share", format=".1%"),
                 alt.Tooltip("paths:Q", title="Paths", format=",")],
    )

    labels_layer = alt.Chart(df).mark_text(
        align="left", dx=6, fontSize=11, color=LABEL
    ).encode(y=alt.Y("when:N", sort=None), x="share:Q",
             text=alt.Text("share:Q", format=".0%"))

    return _finish(bars + labels_layer, height)


def vol_sensitivity(rows, floor, height=300):
    """
    Structuring margin against implied volatility, with the floor drawn.

    The slope is the message. On this structure the margin moves about a
    point of nominal per point of volatility, so where the curve crosses
    the floor decides whether the trade exists at all.
    """
    df = pd.DataFrame(rows)
    df["vol_pct"] = df["vol"] * 100
    df["margin_pct"] = df["margin"] * 100

    line = alt.Chart(df).mark_line(
        strokeWidth=2.5, color=PROTECTED,
        point=alt.OverlayMarkDef(size=70, color=PROTECTED),
    ).encode(
        x=alt.X("vol_pct:Q", title="Implied volatility (%)",
                scale=alt.Scale(zero=False), axis=_axis()),
        y=alt.Y("margin_pct:Q", title="Gross structuring margin (%)",
                axis=_axis()),
        tooltip=[alt.Tooltip("vol_pct:Q", title="Vol", format=".1f"),
                 alt.Tooltip("margin_pct:Q", title="Margin", format=".2f")],
    )

    floor_rule = alt.Chart(pd.DataFrame({"y": [floor * 100]})).mark_rule(
        color=AT_RISK, strokeDash=[5, 4], strokeWidth=1.5
    ).encode(y="y:Q")

    floor_text = alt.Chart(pd.DataFrame({
        "y": [floor * 100], "t": [f"Margin floor {floor:.2%}"],
    })).mark_text(align="left", dy=-8, fontSize=11,
                  color=AT_RISK).encode(y="y:Q", text="t:N",
                                              x=alt.value(8))

    return _finish(floor_rule + line + floor_text, height)


def stepdown_schedule(schedule, height=240):
    """
    Autocall trigger at each call date, where it steps down over time.

    Worth drawing only when the trigger actually moves: a flat line adds
    nothing a sentence could not say.
    """
    times, levels = schedule.call_levels()
    df = pd.DataFrame({"t": times, "level": levels * 100})

    line = alt.Chart(df).mark_line(
        strokeWidth=2.5, color=PROTECTED, interpolate="step-after",
        point=alt.OverlayMarkDef(size=70, color=PROTECTED),
    ).encode(
        x=alt.X("t:Q", title="Years from trade date", axis=_axis()),
        y=alt.Y("level:Q", title="Autocall trigger (% of initial)",
                scale=alt.Scale(zero=False), axis=_axis()),
        tooltip=[alt.Tooltip("t:Q", title="Year", format=".2f"),
                 alt.Tooltip("level:Q", title="Trigger", format=".1f")],
    )

    labels = alt.Chart(df).mark_text(
        dy=-12, fontSize=11, color=LABEL
    ).encode(x="t:Q", y="level:Q", text=alt.Text("level:Q", format=".0f"))

    return _finish(line + labels, height)
